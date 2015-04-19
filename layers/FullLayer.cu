#include"FullLayer.cuh"

__device__ double fActiveFun(double src,int type)
{
	if(type==NL_RELU)
	{
		if(src<0.0)
			return 0.0;
		else
			return src;
	}
	else if(type==NL_SOFT_PLUS)
		return ::log(1+::exp(src));
	return 1/(1+::exp(-src));
}

__device__ double d_fActiveFun(double src,int type)
{
	if(type==NL_RELU)
	{
		if(src>0.0)
			return 1.0;
		else
			return 0.0;

	}
	else if(type==NL_SOFT_PLUS)
		return 1/(1+::exp(-src));
	return src*(1-src);
}

int CFullLayer::initMem()
{
	m_weightLen=m_curNumFeature*m_inputNumFeature;

	cudaError_t cudaStat;

	cudaStat=cudaMalloc((void**)&m_delta,sizeof(double)*m_inputNumFeature*batch);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMalloc failed\n");  
		freeMem();
		return -1;
	}

	cudaStat=cudaMalloc((void**)&m_fullDelta,sizeof(double)*m_curNumFeature*batch);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMalloc failed\n");  
		freeMem();
		return -1;
	}

	cudaStat=cudaMalloc((void**)&m_afterDropWeight,sizeof(double)*m_inputNumFeature*m_curNumFeature);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMalloc failed\n");  
		freeMem();
		return -1;
	}

	cudaStat=cudaMalloc((void**)&m_fullData,sizeof(double)*m_curNumFeature*batch);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMalloc failed\n");  
		freeMem();
		return -1;
	}

	cudaStat=cudaMalloc((void**)&m_fullNoActiveData,sizeof(double)*m_curNumFeature*batch);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMalloc failed\n");  
		freeMem();
		return -1;
	}

	cudaStat=cudaMalloc((void**)&m_dropRate,sizeof(float)*m_inputNumFeature*m_curNumFeature);
	if(cudaStat!=cudaSuccess)
	{
		printf("device memory cudaMemset failed\n");
		freeMem();
		return -1;
	}

	curandStatus_t status=curandCreateGenerator(&m_hGen, CURAND_RNG_PSEUDO_DEFAULT);
	if(status!=CURAND_STATUS_SUCCESS)
	{
		printf("curandCreateGenerator error");
		exit(0);
	}
	status=curandSetPseudoRandomGeneratorSeed(m_hGen,time(NULL));
	if(status!=CURAND_STATUS_SUCCESS)
	{
		printf("curandSetPseudoRandomGeneratorSeed error");
		exit(0);
	}

	return CLayer::initMem();
}

void CFullLayer::freeMem()
{
	GPU_FREE(m_fullData);
	GPU_FREE(m_fullNoActiveData);
	GPU_FREE(m_dropRate);
	GPU_FREE(m_afterDropWeight);
	GPU_FREE(m_fullDelta);

	curandDestroyGenerator(m_hGen);

	CLayer::freeMem();
}

//block<<<weightLeng/threadNum>>>
//thread<<<min(1024,weightLeng)>>>
__global__ void dropOperator(float* dropRate,int leng,double rate)
{
	for(int i = 0; i < leng; i += blockDim.x * gridDim.x) 
	{ 
		int id = i + blockIdx.x * blockDim.x + threadIdx.x; 
		if(id < leng) 
		{
			if(rate==0)
				dropRate[id]=1;
			else if(dropRate[id]>rate)
				dropRate[id]=1;
			else
				dropRate[id]=0;
		}
	}
}

//block<<<weightLeng/threadNum>>>
//thread<<<min(1024,weightLeng)>>>
__global__ void weightOperator(double* afterWeight,double* weight,float* dropRate,int wLeng,double rate,bool bPre)
{
	for(int i = 0; i < wLeng; i += blockDim.x * gridDim.x) 
	{ 
		int id = i + blockIdx.x * blockDim.x + threadIdx.x; 
		if(id < wLeng) 
		{
			if(bPre)
				afterWeight[id]=weight[id]*(1-rate);
			else
				afterWeight[id]=weight[id]*dropRate[id];
		}
	}
}


//block<<<batch>>>
//thread<<<min(1024,numFeature)>>>
__global__ void feedForwardActive(double* activeData,double*notActiveData,double* bias,int numFeature,int activeType)
{
	int index=blockIdx.x*numFeature;
	for(int id = 0; id < numFeature; id += blockDim.x) 
	{ 
		int idx = id + threadIdx.x; 
		if(idx < numFeature)
			activeData[index+idx]=fActiveFun(notActiveData[index+idx]+bias[idx],activeType);
	}
}

void CFullLayer::feedforward(double* srcData,int activeType,bool bPred)
{
	if(curandGenerateUniform(m_hGen,m_dropRate,m_curNumFeature*m_inputNumFeature)!=CURAND_STATUS_SUCCESS)
	{
		printf("curandGenerateUniform error");
		exit(0);
	}
	
	int wLen=m_curNumFeature*m_inputNumFeature;
	dim3 threads = min(1024, wLen); 
	dim3 blocks  = min(1024, (wLen + threads.x - 1) / threads.x); 


	dropOperator<<<blocks,threads>>>(m_dropRate,wLen,m_rate);
	cudaDeviceSynchronize();

	weightOperator<<<blocks,threads>>>(m_afterDropWeight,m_weight,m_dropRate,wLen,m_rate,bPred);
	cudaDeviceSynchronize();

	matrixMulTA(srcData,batch,m_inputNumFeature,m_afterDropWeight,m_curNumFeature,m_inputNumFeature,m_fullNoActiveData,m_curNumFeature);

	blocks=batch;
	threads=min(1024,m_curNumFeature);
	feedForwardActive<<<blocks,threads>>>(m_fullData,m_fullNoActiveData,m_bias,m_curNumFeature,activeType);
	cudaDeviceSynchronize();
}


double CFullLayer::getCost()
{
	int dataSize=m_curNumFeature*m_inputNumFeature;
	double finSum=getWeightCost(m_weight,dataSize);
	return finSum*m_lambda/2;
}



//blocks<<<batch,NumFeature>>>
//thread<<<inputNumFeature>>>
__global__ void dFullActive(
						  double* deltaData,
						  double* deltaDataD,
						  double* fullNoActiveData,
						  int type
						  )
{
	int srcNo=blockIdx.x;
	int featureNo=threadIdx.x;
	int index=srcNo*blockDim.x+featureNo;
	deltaDataD[index]=deltaData[index]*d_fActiveFun(fullNoActiveData[index],type);
}

int CFullLayer::backpropagation(double* nexDeltas,int activeType)
{

	dFullActive<<<batch,m_curNumFeature>>>(nexDeltas,
											  m_fullDelta,
											  m_fullNoActiveData,
										      activeType);
	cudaDeviceSynchronize();

	//因为CUDA以列为主，所以我这里把传入的x,y调换，保证结果以行为主
	matrixMul(m_fullDelta,batch,m_curNumFeature,m_afterDropWeight,m_curNumFeature,m_inputNumFeature,m_delta,m_inputNumFeature);
	return 0;
}

void CFullLayer::getGrad(double* srcData)
{

	matrixMulTB(m_fullDelta,m_curNumFeature,srcData,batch,m_inputNumFeature,m_weightGrad,m_inputNumFeature);

	dim3 blocks(m_curNumFeature,m_inputNumFeature);

	int threadNum=min(MAX_THREAD_NUM,batch);

	int wLen=m_curNumFeature*m_inputNumFeature;
	dim3 threads2 = min(1024, wLen); 
	dim3 blocks2  = min(1024, (wLen + threads2.x - 1) / threads2.x); 

	fullWeightGrad2<<<blocks2,threads2>>>(m_weightGrad,m_weight,m_dropRate,wLen,m_lambda,batch);

	cudaDeviceSynchronize();

	fullBiasGrad<<<m_curNumFeature,threadNum,sizeof(double)*threadNum>>>(m_fullDelta,
							   m_biasGrad,
							   batch);
	cudaDeviceSynchronize();
}

void CFullLayer::updateWeight(float mom,float alpha)
{
	int threadNum=min(1024,m_inputNumFeature);
	g_weightAndBiasAdd<<<m_curNumFeature,threadNum>>>(m_weight,m_weightGrad,
												      m_vecWeight,m_bias,
												      m_biasGrad,m_vecBias,
												      m_inputNumFeature,
												      mom,alpha);

	cudaDeviceSynchronize();
}