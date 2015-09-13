#include"softmaxLayer.cuh"

int CSoftMaxLayerGPU::initMem()
{

	m_weightLen=m_curNumFeature*m_inputNumFeature;
	DL_ASSER(m_weightLen!=0);

	cudaError_t cudaStat;

	cudaStat=cudaMalloc((void**)&m_fullData,sizeof(double)*m_curNumFeature*batch);
	CUDA_ERROR(cudaStat);

	cudaStat=cudaMalloc((void**)&m_delta,sizeof(double)*m_curNumFeature*batch);
	CUDA_ERROR(cudaStat);

	m_truthData=new double[m_curNumFeature*batch];
	DL_ASSER(m_truthData!=NULL);

	return CLayerGPU::initMem();
}


void CSoftMaxLayerGPU::freeMem()
{
	GPU_FREE(m_fullData);
	if(m_truthData)
	{
		delete [] m_truthData;
		m_truthData=NULL;
	}
	CLayerGPU::freeMem();
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

void CSoftMaxLayerGPU::feedforward(double*srcData,DLparam& params)
{
	bool bPred=params.pred;
	matrixMulTA(srcData,batch,m_inputNumFeature,m_weight,m_curNumFeature,m_inputNumFeature,m_fullData,m_curNumFeature);

	cudaError_t cudaStat;
	dim3 blocks=batch;
	dim3 threads=min(1024,m_curNumFeature);
	feedForwardAddBias<<<blocks,threads>>>(m_fullData,m_curNumFeature,m_bias);
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	//cpu计算最大值
	cudaStat=cudaMemcpy(m_truthData,m_fullData,sizeof(double)*m_curNumFeature*batch,cudaMemcpyDeviceToHost);
	CUDA_ERROR(cudaStat);

	double * tmpMax=new double[batch];

	for(int i=0;i<batch;i++)
	{
		double max=0.0;
		for(int j=0;j<m_curNumFeature;j++)
		{
			if(m_truthData[i*m_curNumFeature+j]>max)
				max=m_truthData[i*m_curNumFeature+j];
		}
		tmpMax[i]=max;
	}

	//计算概率
	for(int i=0;i<batch;i++)
	{
		double tSum=0.0;
		for(int j=0;j<m_curNumFeature;j++)
		{
			m_truthData[i*m_curNumFeature+j]-=tmpMax[i];
			m_truthData[i*m_curNumFeature+j]=exp(m_truthData[i*m_curNumFeature+j]);
			tSum+=m_truthData[i*m_curNumFeature+j];
		}
		for(int j=0;j<m_curNumFeature;j++)
			m_truthData[i*m_curNumFeature+j]/=tSum;
	}

	delete [] tmpMax;

	if(bPred)
	{
		for(int i=0;i<batch;i++)
		{
			double max=0.0;
			int index=0;
			for(int j=0;j<m_curNumFeature;j++)
			{
				if(m_truthData[i*m_curNumFeature+j]>max)
				{
					index=j;
					max=m_truthData[i*m_curNumFeature+j];
				}
			}
			params.predData[i]=index;
		}
	}
	else
	{
		cudaStat=cudaMemcpy(m_fullData,m_truthData,sizeof(double)*m_curNumFeature*batch,cudaMemcpyHostToDevice);
		CUDA_ERROR(cudaStat);
	}
}

double  CSoftMaxLayerGPU::getCost(DLparam& params)
{
	int dataSize=m_curNumFeature*m_inputNumFeature;
	double finSum=getWeightCost(m_weight,dataSize);

	cudaError_t cudaStat;
	cudaStat=cudaMemcpy(m_truthData,m_fullData,sizeof(double)*m_curNumFeature*batch,cudaMemcpyDeviceToHost);
	CUDA_ERROR(cudaStat);

	double preSum=0.0;
	for(int i=0;i<batch;i++)
		preSum+=log(m_truthData[i*m_curNumFeature+params.labels[i]]);

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

void CSoftMaxLayerGPU::backpropagation(double*preDelta,DLparam& params)
{
	memset(m_truthData,0,sizeof(double)*m_curNumFeature*batch);
	for(int i=0;i<params.labels.size();i++)
		m_truthData[params.labels[i]+i*m_curNumFeature]=1;

	cudaError_t cudaStat=cudaMemcpy(m_delta,m_truthData,sizeof(double)*m_curNumFeature*batch,cudaMemcpyHostToDevice);
	CUDA_ERROR(cudaStat);

	dim3 blocks(batch,m_curNumFeature);
	softmaxGetTruthDelta<<<blocks,1>>>(m_delta,m_fullData);
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	cudaStat=cudaMemset(preDelta,0,sizeof(double)*batch*m_inputNumFeature);
	CUDA_ERROR(cudaStat);

	matrixMul(m_delta,batch,m_curNumFeature,m_weight,m_curNumFeature,m_inputNumFeature,preDelta,m_inputNumFeature);
}

void  CSoftMaxLayerGPU::getGrad(double* srcData)
{
	matrixMulTB(m_delta,m_curNumFeature,srcData,batch,m_inputNumFeature,m_weightGrad,m_inputNumFeature);

	dim3 blocks(m_curNumFeature,m_inputNumFeature);

	cudaError_t cudaStat;
	int threadNum=min(MAX_THREAD_NUM,batch);

	fullWeightGrad<<<blocks,1>>>(m_weightGrad,m_weight,m_lambda,batch);
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	dim3 block2(m_curNumFeature);
	fullBiasGrad<<<m_curNumFeature,threadNum,sizeof(double)*threadNum>>>(m_delta,
							   m_biasGrad,
							   batch);
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);
}
 
void CSoftMaxLayerGPU::updateWeight(float mom,float alpha)
{
	int threadNum=min(1024,m_inputNumFeature);
	g_weightAndBiasAdd<<<m_curNumFeature,threadNum>>>(m_weight,m_weightGrad,
												      m_vecWeight,m_bias,
												      m_biasGrad,m_vecBias,
												      m_inputNumFeature,
												      mom,alpha);
	cudaError_t cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);
}