#include"PoolLayer.cuh"

int CPoolLayerGPU::initMem()
{
	DL_ASSER(m_preConvDim != 0);

	unsigned const int convDataSize=sizeof(double)*m_preConvDim*m_preConvDim*batch*m_curNumFeature;

	cudaError_t cudaStat=cudaMalloc((void**)&m_delta,convDataSize/m_kernelDim/m_kernelDim);
	CUDA_ERROR(cudaStat);

	cudaStat=cudaMalloc((void**)&m_poolData,convDataSize/m_kernelDim/m_kernelDim);
	CUDA_ERROR(cudaStat);

	cudaStat=cudaMalloc((void**)&m_maxIndexData,sizeof(int)*m_preConvDim*m_preConvDim*batch*m_curNumFeature/m_kernelDim/m_kernelDim);
	CUDA_ERROR(cudaStat);

	return 0;
}

void CPoolLayerGPU::freeMem()
{
	GPU_FREE(m_delta);
	GPU_FREE(m_poolData);
	GPU_FREE(m_maxIndexData);

	CLayerGPU::freeMem();
}


//block<<<batch,m_curNumFeature>>>
//threads<<<min(1024,poolDim*poolDim)>>>
__global__ void maxPool(
						double* convData,
						int convDim,
						int numFeature,
						double* poolData,
						int* maxIndexData,
						int poolArea
						)
{
	int srcNo=blockIdx.x;
	int featureNo=blockIdx.y;
	int conv2=convDim*convDim;
	int poolDim=convDim/poolArea;
	int poolDim2=poolDim*poolDim;
	double* pConv=convData+(srcNo*numFeature+featureNo)*conv2;
	double* pPoolData=poolData+(srcNo*numFeature+featureNo)*poolDim2;
	int*    pIndex=maxIndexData+(srcNo*numFeature+featureNo)*poolDim2;

	for(int idx=threadIdx.x;idx<poolDim2;idx+=blockDim.x)
	{
		int convX=(idx%poolDim)*poolArea;
		int convY=(idx/poolDim)*poolArea;

		double maxNum=0.0;
		double dTmp=0.0;
		int index=0;

		for(int poolRow =0;poolRow<poolArea;poolRow++)
		{
			for(int poolCol = 0;poolCol<poolArea;poolCol++)
			{
				dTmp=pConv[(convY+poolRow)*convDim+convX+poolCol];
				if(maxNum<dTmp)
				{
					maxNum=dTmp;
					index=poolRow*poolArea+poolCol;
				}
			}
		}
		pIndex[threadIdx.y*poolDim+threadIdx.x]=index;
		pPoolData[threadIdx.y*poolDim+threadIdx.x]=maxNum;
	}
}

//block<<<batch,m_curNumFeature>>>
//threads<<<min(1024,poolDim*poolDim)>>>
__global__ void avgPool(
						double* convData,
						int convDim,
						int numFeature,
						double* poolData,
						int poolArea
						)
{
	int srcNo=blockIdx.x;
	int featureNo=blockIdx.y;
	int conv2=convDim*convDim;
	int poolDim=convDim/poolArea;
	int poolDim2=poolDim*poolDim;
	double* pConv=convData+(srcNo*numFeature+featureNo)*conv2;
	double* pPoolData=poolData+(srcNo*numFeature+featureNo)*poolDim2;

	for(int idx=threadIdx.x;idx<poolDim2;idx+=blockDim.x)
	{
		int convX=(idx%poolDim)*poolArea;
		int convY=(idx/poolDim)*poolArea;

		double avgNum=0.0;

		for(int poolRow =0;poolRow<poolArea;poolRow++)
		{
			for(int poolCol = 0;poolCol<poolArea;poolCol++)
				avgNum+=pConv[(convY+poolRow)*convDim+convX+poolCol];
		}
		avgNum/=poolArea*poolArea;
		pPoolData[threadIdx.y*poolDim+threadIdx.x]=avgNum;
	}
}

void CPoolLayerGPU::feedforward(double* srcData,DLparam& params)
{
	dim3 blocks(batch,m_curNumFeature);
	int poolDim=m_preConvDim/m_kernelDim;
	dim3 threads=min(1024,poolDim*poolDim);
	if(m_poolType==MAX_POOL)
		maxPool<<<blocks,threads>>>
							(srcData,
							 m_preConvDim,
							 m_curNumFeature,
							 m_poolData,
							 m_maxIndexData,
							 m_kernelDim);
	else
		avgPool<<<blocks,threads>>>
							(srcData,
							 m_preConvDim,
							 m_curNumFeature,
							 m_poolData,
							 m_kernelDim);
	cudaError_t cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);
}

//block<<<batch,m_curNumFeature>>>
//threads<<<min(1024,poolDim*poolDim)>>>
__global__ void backDeltaFromMaxPoolToConv(
						double* convDelta,
						int convDim,
						int numFeature,
						double* poolDelta,
						int* maxIndexData,
						int poolArea
						)
{
	int srcNo=blockIdx.x;
	int featureNo=blockIdx.y;
	int conv2=convDim*convDim;
	int poolDim=convDim/poolArea;
	int poolDim2=poolDim*poolDim;
	double* pConvDelta=convDelta+(srcNo*numFeature+featureNo)*conv2;
	double* pPoolData=poolDelta+(srcNo*numFeature+featureNo)*poolDim2;
	int*    pIndex=maxIndexData+(srcNo*numFeature+featureNo)*poolDim2;

	for(int idx=threadIdx.x;idx<poolDim2;idx+=blockDim.x)
	{
		int convX=(idx%poolDim)*poolArea;
		int convY=(idx/poolDim)*poolArea;
		double dTmp=0.0;

		dTmp=pPoolData[idx];
		int index=pIndex[idx];
		int row=index/poolArea;
		int col=index%poolArea;
		pConvDelta[(convY+row)*convDim+convX+col]=dTmp;
	}
}

//block<<<batch,m_curNumFeature>>>
//threads<<<min(1024,convDim*convDim)>>>
__global__ void backDeltaFromAvgPoolToConv(
						double* convDelta,
						int convDim,
						int numFeature,
						double* poolDelta,
						int poolArea
						)
{
	int srcNo=blockIdx.x;
	int featureNo=blockIdx.y;
	int conv2=convDim*convDim;
	int poolDim=convDim/poolArea;
	int poolDim2=poolDim*poolDim;
	int poolArea2=poolArea*poolArea;
	double* pConvDelta=convDelta+(srcNo*numFeature+featureNo)*conv2;
	double* pPoolData=poolDelta+(srcNo*numFeature+featureNo)*poolDim2;

	for(int idx=threadIdx.x;idx<conv2;idx+=blockDim.x)
	{
		int poolX=(idx%convDim)/poolArea;
		int poolY=(idx/convDim)/poolArea;
		pConvDelta[idx]=pPoolData[poolY*poolDim+poolX]/poolArea2;
	}
}

void CPoolLayerGPU::backpropagation(double* preDelta,DLparam& params)
{
	cudaError_t cudaStat;

	cudaStat=cudaMemset(preDelta,0,sizeof(double)*m_preConvDim*m_preConvDim*batch*m_curNumFeature);
	CUDA_ERROR(cudaStat);

	dim3 blocks(batch,m_curNumFeature);
	int poolDim=m_preConvDim/m_kernelDim;
	dim3 threads(min(1024,poolDim*poolDim));

	if(m_poolType==MAX_POOL)
		backDeltaFromMaxPoolToConv<<<blocks,threads>>>(preDelta,
												   m_preConvDim,
												   m_curNumFeature,
												   m_delta,
												   m_maxIndexData,
												   m_kernelDim
												   );
	else
	{
		threads=min(1024,m_preConvDim*m_preConvDim);
		backDeltaFromAvgPoolToConv<<<blocks,threads>>>(preDelta,
													   m_preConvDim,
													   m_curNumFeature,
													   m_delta,
												       m_kernelDim
												       );
	}
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);
}