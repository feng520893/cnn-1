#include"PoolLayer.cuh"

int CPoolLayerGPU::setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs)
{
	PoolParam poolParam=m_param.poolParam;

	int dataHeightDim=ceil((float)inputs[0]->dimHeight/poolParam.kernelDim);
	int dataWidthDim=ceil((float)inputs[0]->dimWidth/poolParam.kernelDim);
	outputs[0]->create(inputs[0]->num,inputs[0]->dataChannel,dataHeightDim,dataWidthDim);
	m_maxIndexData.create(inputs[0]->num,inputs[0]->dataChannel,dataHeightDim,dataWidthDim);

	return 0;
}


//block<<<batch,m_curNumFeature>>>
//threads<<<min(1024,poolDim*poolDim)>>>
__global__ void maxPool(
						precision* convData,
						int convDim,
						int numFeature,
						precision* poolData,
						int poolDim,
						precision* maxIndexData,
						int poolArea
						)
{
	int srcNo=blockIdx.x;
	int featureNo=blockIdx.y;
	int conv2=convDim*convDim;
	int poolDim2=poolDim*poolDim;
	precision* pConv=convData+(srcNo*numFeature+featureNo)*conv2;
	precision* pPoolData=poolData+(srcNo*numFeature+featureNo)*poolDim2;
	precision* pIndex=maxIndexData+(srcNo*numFeature+featureNo)*poolDim2;

	for(int idx=threadIdx.x;idx<poolDim2;idx+=blockDim.x)
	{
		int convX=(idx%poolDim)*poolArea;
		int convY=(idx/poolDim)*poolArea;

		precision maxNum=0.0;
		precision dTmp=0.0;
		int index=0;

		for(int poolRow =0;poolRow<poolArea&&convY+poolRow<convDim;poolRow++)
		{
			for(int poolCol = 0;poolCol<poolArea&&convX+poolCol<convDim;poolCol++)
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
						precision* convData,
						int convDim,
						int numFeature,
						precision* poolData,
						int poolDim,
						int poolArea
						)
{
	int srcNo=blockIdx.x;
	int featureNo=blockIdx.y;
	int conv2=convDim*convDim;
	int poolDim2=poolDim*poolDim;
	precision* pConv=convData+(srcNo*numFeature+featureNo)*conv2;
	precision* pPoolData=poolData+(srcNo*numFeature+featureNo)*poolDim2;

	for(int idx=threadIdx.x;idx<poolDim2;idx+=blockDim.x)
	{
		int convX=(idx%poolDim)*poolArea;
		int convY=(idx/poolDim)*poolArea;

		precision avgNum=0.0;

		for(int poolRow =0;poolRow<poolArea&&convY+poolRow<convDim;poolRow++)
			for(int poolCol = 0;poolCol<poolArea&&convX+poolCol<convDim;poolCol++)
				avgNum+=pConv[(convY+poolRow)*convDim+convX+poolCol];
		avgNum/=poolArea*poolArea;
		pPoolData[threadIdx.y*poolDim+threadIdx.x]=avgNum;
	}
}

precision CPoolLayerGPU::feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	dim3 blocks(bottoms[0]->num,bottoms[0]->dataChannel);
	int poolDim=tops[0]->dimWidth;
	dim3 threads=min(1024,poolDim*poolDim);
	if(m_param.poolParam.poolType==MAX_POOL)
		maxPool<<<blocks,threads>>>
						(bottoms[0]->gpuData,
						 bottoms[0]->dimWidth,
						 bottoms[0]->dataChannel,
						 tops[0]->gpuData,
						 tops[0]->dimWidth,
						 m_maxIndexData.gpuData,
						 m_param.poolParam.kernelDim);
	else
		avgPool<<<blocks,threads>>>
						(bottoms[0]->gpuData,
						 bottoms[0]->dimWidth,
						 bottoms[0]->dataChannel,
						 tops[0]->gpuData,
						 tops[0]->dimWidth,
						 m_param.poolParam.kernelDim);
	cudaError_t cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);
	return 0;
}

//block<<<batch,m_curNumFeature>>>
//threads<<<min(1024,poolDim*poolDim)>>>
__global__ void backDeltaFromMaxPoolToConv(
						precision* convDelta,
						int convDim,
						int numFeature,
						precision* poolDelta,
						int poolDim,
						precision* maxIndexData,
						int poolArea
						)
{
	int srcNo=blockIdx.x;
	int featureNo=blockIdx.y;
	int conv2=convDim*convDim;
	int poolDim2=poolDim*poolDim;
	precision* pConvDelta=convDelta+(srcNo*numFeature+featureNo)*conv2;
	precision* pPoolData=poolDelta+(srcNo*numFeature+featureNo)*poolDim2;
	precision* pIndex=maxIndexData+(srcNo*numFeature+featureNo)*poolDim2;

	for(int idx=threadIdx.x;idx<poolDim2;idx+=blockDim.x)
	{
		int convX=(idx%poolDim)*poolArea;
		int convY=(idx/poolDim)*poolArea;
		precision dTmp=0.0;

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
						precision* convDelta,
						int convDim,
						int numFeature,
						precision* poolDelta,
						int poolDim,
						int poolArea
						)
{
	int srcNo=blockIdx.x;
	int featureNo=blockIdx.y;
	int conv2=convDim*convDim;
	int poolDim2=poolDim*poolDim;
	int poolArea2=poolArea*poolArea;
	precision* pConvDelta=convDelta+(srcNo*numFeature+featureNo)*conv2;
	precision* pPoolData=poolDelta+(srcNo*numFeature+featureNo)*poolDim2;

	for(int idx=threadIdx.x;idx<conv2;idx+=blockDim.x)
	{
		int poolX=(idx%convDim)/poolArea;
		int poolY=(idx/convDim)/poolArea;
		pConvDelta[idx]=pPoolData[poolY*poolDim+poolX]/poolArea2;
	}
}

int CPoolLayerGPU::backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms)
{
	cudaError_t cudaStat;
	
	cudaStat=cudaMemset(bottoms[0]->gpuDiff,0,sizeof(precision)*bottoms[0]->size());
	CUDA_ERROR(cudaStat);

	dim3 blocks(tops[0]->num,tops[0]->dataChannel);
	int poolDim=tops[0]->dimWidth;
	dim3 threads(min(1024,poolDim*poolDim));

	if(m_param.poolParam.poolType==MAX_POOL)
		backDeltaFromMaxPoolToConv<<<blocks,threads>>>(bottoms[0]->gpuDiff,
													   bottoms[0]->dimWidth,
												       bottoms[0]->dataChannel,
												       tops[0]->gpuDiff,
													   tops[0]->dimWidth,
												       m_maxIndexData.gpuData,
												       m_param.poolParam.kernelDim
												       );
	else
	{
		threads=min(1024,bottoms[0]->dimWidth*bottoms[0]->dimWidth);
		backDeltaFromAvgPoolToConv<<<blocks,threads>>>(bottoms[0]->gpuDiff,
													   bottoms[0]->dimWidth,
													   bottoms[0]->dataChannel,
													   tops[0]->gpuDiff,
													   tops[0]->dimWidth,
												       m_param.poolParam.kernelDim
												      );
	}
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);
	return NET_SUCCESS;
}