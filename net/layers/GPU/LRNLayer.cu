#include"LRNLayer.cuh"

int CLRNLayerGPU::setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs)
{
	DL_ASSER(m_param.lrnParam.localSize%2!=0);
	m_prePad = (m_param.lrnParam.localSize-1)/2;
	if(m_param.lrnParam.normRegionType == ACROSS_CHANNELS)
	{
		m_scales.create(inputs[0]->num,inputs[0]->dataChannel,inputs[0]->dimHeight,inputs[0]->dimWidth);
		outputs[0]->create(inputs[0]->num,inputs[0]->dataChannel,inputs[0]->dimHeight,inputs[0]->dimWidth);
	}
	return 0;
}

//block<<<batch,channels>>>
//threads<<<min(1024,dimHeight*dimWidth)>>>
__global__ void computScales(
						precision* scales,
						precision* inputs,
						unsigned int dataSize,
						int dimSize,
						int padded,
						int localSize,
						precision alphaOver,
						int k=1
						)
{
	unsigned int numOffset=blockIdx.x*dataSize;
	int nChannelId=blockIdx.y;
	unsigned int tid=threadIdx.x;
	
	int first=max(0,nChannelId-padded);
	int end=min(nChannelId+localSize-padded,gridDim.y);

	for(unsigned int i=tid;i<dimSize;i+=blockDim.x)
	{
		precision tmp=0.0;
		for(int c=first;c<end;c++)
			tmp+=pow(inputs[numOffset+c*dimSize+i],2);
		scales[numOffset+nChannelId*dimSize+i]=k+tmp*alphaOver;
	}
}

//block<<<weightLeng/threadNum>>>
//thread<<<min(1024,weightLeng)>>>
__global__ void normalAcrossChannels(precision* inputDatas, precision* scales,precision* normalDatas,unsigned int dataLeng,precision beat) 
{ 
	for(int i = 0; i < dataLeng; i += blockDim.x * gridDim.x) 
	{ 
		int id = i + blockIdx.x * blockDim.x + threadIdx.x; 
		if(id < dataLeng) 
			normalDatas[id] = inputDatas[id]*pow(scales[id],-beat); 
	}
}

precision CLRNLayerGPU::feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	int nRet=NET_SUCCESS;
	if(m_param.lrnParam.normRegionType == ACROSS_CHANNELS)
		nRet=crossChannelForward(bottoms,tops);
	return nRet;
}

int CLRNLayerGPU::crossChannelForward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	int localSize=m_param.lrnParam.localSize;
	precision alphaOver=m_param.lrnParam.alpha/localSize;
	
	unsigned int dataSize=bottoms[0]->size()/bottoms[0]->num;
	int dimSize=bottoms[0]->dimHeight*tops[0]->dimWidth;

	dim3 blocks(bottoms[0]->num,bottoms[0]->dataChannel);
	dim3 threads=min(1024,dimSize);
	computScales<<<blocks,threads>>>(m_scales.gpuData,
									 bottoms[0]->gpuData,
									 dataSize,
									 dimSize,
									 m_prePad,
									 m_param.lrnParam.localSize,
									 alphaOver);
	cudaError_t cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	unsigned int leng=m_scales.size();
	threads= min(1024,leng);
	blocks = min(65535, (leng + threads.x - 1) / threads.x); 

	normalAcrossChannels<<<blocks,threads>>>(bottoms[0]->gpuData,
		                                     m_scales.gpuData,
											 tops[0]->gpuData,
											 leng,
											 m_param.lrnParam.beat
											 );
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	return NET_SUCCESS;
}

int CLRNLayerGPU::backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms)
{
	int nRet=NET_SUCCESS;
	if(m_param.lrnParam.normRegionType == ACROSS_CHANNELS)
		nRet=crossChannelBack(tops,propagateDown,bottoms);
	return nRet;
}

//block<<<batch,channels>>>
//threads<<<min(1024,dimHeight*dimWidth)>>>
__global__ void computBottomDiffs(
						precision* topsData,
						precision* topsDiff,
						precision* scales,
						precision* bottomsData,
						precision* bottomsDiff,
						unsigned int dataSize,
						int dimSize,
						int padded,
						int localSize,
						precision cacheRatioValue
						)
{
	unsigned int numOffset=blockIdx.x*dataSize;
	int nChannelId=blockIdx.y;
	unsigned int tid=threadIdx.x;
	
	int first=max(0,nChannelId-padded);
	int end=min(nChannelId+localSize-padded,gridDim.y);

	for(unsigned int i=tid;i<dimSize;i+=blockDim.x)
	{
		precision tmp=0.0;
		for(int c=first;c<end;c++)
		{
			int offset=numOffset+c*dimSize+i;
			tmp+=topsDiff[offset]*topsData[offset]/scales[offset];
		}
		bottomsDiff[numOffset+nChannelId*dimSize+i]-=cacheRatioValue*(bottomsData[numOffset+nChannelId*dimSize+i]*tmp);
	}
}


int CLRNLayerGPU::crossChannelBack(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms)
{
	int channels=bottoms[0]->dataChannel;
	precision beat=m_param.lrnParam.beat;
	int localSize=m_param.lrnParam.localSize;
	precision cacheRatioValue = 2 * m_param.lrnParam.alpha * beat / localSize;
	unsigned int dataSize=bottoms[0]->size()/bottoms[0]->num;
	int dimSize=bottoms[0]->dimHeight*bottoms[0]->dimWidth;
	unsigned int leng=m_scales.size();
	dim3 threads = min(1024,leng);
	dim3 blocks = min(65535,(leng + threads.x - 1) / threads.x); 

	normalAcrossChannels<<<blocks,threads>>>(tops[0]->gpuDiff,
		                                     m_scales.gpuData,
											 bottoms[0]->gpuDiff,
											 leng,
											 m_param.lrnParam.beat
											 );
	cudaError_t cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	blocks=dim3(bottoms[0]->num,bottoms[0]->dataChannel);
	threads=min(1024,dimSize);
	computBottomDiffs<<<blocks,threads>>>(tops[0]->gpuData,
		                                  tops[0]->gpuDiff,
										  m_scales.gpuData,
									      bottoms[0]->gpuData,
										  bottoms[0]->gpuDiff,
										  dataSize,
										  dimSize,
										  m_prePad,
										  localSize,
									      cacheRatioValue);
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	return NET_SUCCESS;
}