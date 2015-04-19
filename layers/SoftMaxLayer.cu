#include"softmaxLayer.cuh"

int CSoftMaxLayer::initMem()
{

	m_weightLen=m_curNumFeature*m_inputNumFeature;

	cudaError_t cudaStat;

	cudaStat=cudaMalloc((void**)&m_fullData,sizeof(double)*m_curNumFeature*batch);
	if(cudaStat!=cudaSuccess)
	{
		printf("device memory cudaMalloc failed\n");  
		freeMem();
		return -1;
	}

	cudaStat=cudaMalloc((void**)&m_gpuTruthDelta,sizeof(double)*m_curNumFeature*batch);
	if(cudaStat!=cudaSuccess)
	{
		printf("device memory cudaMalloc failed\n");  
		freeMem();
		return -1;
	}

	cudaStat=cudaMalloc((void**)&m_delta,sizeof(double)*m_inputNumFeature*batch);
	if(cudaStat!=cudaSuccess)
	{
		printf("device memory cudaMalloc failed\n");  
		freeMem();
		return -1;
	}

	m_truthData=new double[m_curNumFeature*batch];
	return CLayer::initMem();
}


void CSoftMaxLayer::freeMem()
{
	GPU_FREE(m_fullData);
	GPU_FREE(m_gpuTruthDelta);
	if(m_truthData)
	{
		delete [] m_truthData;
		m_truthData=NULL;
	}
	CLayer::freeMem();
}

//block<<<batch>>>
//thread<<<min(1024,numFeature)>>>
__global__ void feedForwardAddBias(double* activeData,int numFeature,double* bias)
{
	int index=blockIdx.x*numFeature;
	for(int id = 0; id < numFeature; id += blockDim.x) 
	{ 
		int idx = id + threadIdx.x; 
		if(idx < numFeature)
			activeData[index+idx]+=bias[idx];
	}
}

int CSoftMaxLayer::feedforward(double** predData,double*srcData,bool bPred)
{
/*	dim3 blocks(batch,m_curNumFeature);
	int threadNum=min(1024,m_inputNumFeature);
	g_fullConnect<<<blocks,threadNum,sizeof(double)*threadNum>>>(
						   srcData,
						   m_weight,
						   m_inputNumFeature,
						   m_fullData,
						   m_bias
						   );

	cudaDeviceSynchronize();*/
	matrixMulTA(srcData,batch,m_inputNumFeature,m_weight,m_curNumFeature,m_inputNumFeature,m_fullData,m_curNumFeature);

	dim3 blocks=batch;
	dim3 threads=min(1024,m_curNumFeature);
	feedForwardAddBias<<<blocks,threads>>>(m_fullData,m_curNumFeature,m_bias);
	cudaDeviceSynchronize();

	//cpu计算最大值
	double * tmp=new double[m_curNumFeature*batch];

	cudaError_t cudaStat=cudaMemcpy(tmp,m_fullData,sizeof(double)*m_curNumFeature*batch,cudaMemcpyDeviceToHost);
	if(cudaStat!=cudaSuccess)
	{
		printf("device memory cudaMempy to host failed\n");  
		freeMem();
		return -3;
	}

	double * tmpMax=new double[batch];

	for(int i=0;i<batch;i++)
	{
		double max=0.0;
		for(int j=0;j<m_curNumFeature;j++)
		{
			if(tmp[i*m_curNumFeature+j]>max)
				max=tmp[i*m_curNumFeature+j];
		}
		tmpMax[i]=max;
	}

	//计算概率
	for(int i=0;i<batch;i++)
	{
		double tSum=0.0;
		for(int j=0;j<m_curNumFeature;j++)
		{
			tmp[i*m_curNumFeature+j]-=tmpMax[i];
			tmp[i*m_curNumFeature+j]=exp(tmp[i*m_curNumFeature+j]);
			tSum+=tmp[i*m_curNumFeature+j];
		}
		for(int j=0;j<m_curNumFeature;j++)
			tmp[i*m_curNumFeature+j]/=tSum;
	}

	delete [] tmpMax;

	if(bPred)
	{
		*predData=new double[batch];
		for(int i=0;i<batch;i++)
		{
			double max=0.0;
			int index=0;
			for(int j=0;j<m_curNumFeature;j++)
			{
				if(tmp[i*m_curNumFeature+j]>max)
				{
					index=j;
					max=tmp[i*m_curNumFeature+j];
				}
			}
			(*predData)[i]=index;
		}
		delete [] tmp;
		return 0;
	}

	cudaMemcpy(m_fullData,tmp,sizeof(double)*m_curNumFeature*batch,cudaMemcpyHostToDevice);
	if(cudaStat!=cudaSuccess)
	{
		printf("device memory cudaMempy to dev failed\n");  
		freeMem();
		return -4;
	}
	delete [] tmp;
	return 0;
}

double  CSoftMaxLayer::getCost(std::vector<int>& labels)
{
	int dataSize=m_curNumFeature*m_inputNumFeature;
	double finSum=getWeightCost(m_weight,dataSize);

	double *tmp=new double[m_curNumFeature*batch];

	cudaMemcpy(tmp,m_fullData,sizeof(double)*m_curNumFeature*batch,cudaMemcpyDeviceToHost);

	double preSum=0.0;
	for(int i=0;i<batch;i++)
		preSum+=log(tmp[i*m_curNumFeature+labels[i]]);
	delete [] tmp;
	tmp=NULL;

	preSum/=-batch;
	return preSum+finSum*m_lambda/2;
}


//block<<<batch,m_curNumFeature>>>
//thread<<<1>>>
__global__ void softmaxGetTruthDelta(double* truthData,double* probData)
{
	int imgNo=blockIdx.x;
	int featureNo=blockIdx.y;
	int index=imgNo*gridDim.y+featureNo;
	truthData[index]=-(truthData[index]-probData[index]);
}

//block<<<batch,m_inputNumFeature>>>
//thread<<<m_curNumFeature>>>
//share<<<sizeof(double)*m_curNumFeature>>>
__global__ void softmaxGetDelta(double* truthDelta,
								double* weight,
								int m_curNumFeature,
								double* delta
								)
{

	extern __shared__ double shared[];

	int imgNo=blockIdx.x;
	int inputFeatureNo=blockIdx.y;
	int featureNo=threadIdx.x;
	int tIndex=imgNo*m_curNumFeature+featureNo;
	int wIndex=featureNo*gridDim.y+inputFeatureNo;
	int dIndex=imgNo*gridDim.y+inputFeatureNo;
	shared[featureNo]=weight[wIndex]*truthDelta[tIndex];
	if(featureNo==0)
	{
		for(int i=0;i<m_curNumFeature;i++)
			delta[dIndex]+=shared[i];
	}
}

double* CSoftMaxLayer::backpropagation(std::vector<int>& labels)
{
	memset(m_truthData,0,sizeof(double)*m_curNumFeature*batch);
	for(int i=0;i<labels.size();i++)
		m_truthData[labels[i]+i*m_curNumFeature]=1;

	cudaError_t cudaStat=cudaMemcpy(m_gpuTruthDelta,m_truthData,sizeof(double)*m_curNumFeature*batch,cudaMemcpyHostToDevice);
	if(cudaStat!=cudaSuccess)
	{
		printf("device memory cudaMemcpy dev failed\n");  
		freeMem();
		return NULL;
	}

	dim3 blocks(batch,m_curNumFeature);
	softmaxGetTruthDelta<<<blocks,1>>>(m_gpuTruthDelta,m_fullData);
	cudaDeviceSynchronize();

	cudaMemset(m_delta,0,sizeof(double)*batch*m_inputNumFeature);
/*
	dim3 blocks2(batch,m_inputNumFeature);
	softmaxGetDelta<<<blocks2,m_curNumFeature,sizeof(double)*m_curNumFeature>>>(m_gpuTruthDelta,
																				m_weight,
																				m_curNumFeature,
																				m_delta);
	cudaDeviceSynchronize();*/
	matrixMul(m_gpuTruthDelta,batch,m_curNumFeature,m_weight,m_curNumFeature,m_inputNumFeature,m_delta,m_inputNumFeature);
	return m_delta;
}

void  CSoftMaxLayer::getGrad(double* srcData)
{
	matrixMulTB(m_gpuTruthDelta,m_curNumFeature,srcData,batch,m_inputNumFeature,m_weightGrad,m_inputNumFeature);

	dim3 blocks(m_curNumFeature,m_inputNumFeature);

	int threadNum=min(MAX_THREAD_NUM,batch);

	fullWeightGrad<<<blocks,1>>>(m_weightGrad,m_weight,m_lambda,batch);
	cudaDeviceSynchronize();

	dim3 block2(m_curNumFeature);
	fullBiasGrad<<<m_curNumFeature,threadNum,sizeof(double)*threadNum>>>(m_gpuTruthDelta,
							   m_biasGrad,
							   batch);
	cudaDeviceSynchronize();

}
 
void CSoftMaxLayer::updateWeight(float mom,float alpha)
{
	int threadNum=min(1024,m_inputNumFeature);
	g_weightAndBiasAdd<<<m_curNumFeature,threadNum>>>(m_weight,m_weightGrad,
												      m_vecWeight,m_bias,
												      m_biasGrad,m_vecBias,
												      m_inputNumFeature,
												      mom,alpha);
	cudaDeviceSynchronize();

}