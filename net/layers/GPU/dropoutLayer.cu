#include"dropoutlayer.cuh"

int CDropoutLayerGPU::setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs)
{
	m_mask.create(inputs[0]->num,inputs[0]->dataChannel,inputs[0]->dimHeight,inputs[0]->dimWidth);
	return NET_SUCCESS;
}

//block<<<weightLeng/threadNum>>>
//thread<<<min(1024,weightLeng)>>>
__global__ void bernoulli(precision* mask,int wLeng,float rate)
{
	for(int i = 0; i < wLeng; i += blockDim.x * gridDim.x) 
	{ 
		int id = i + blockIdx.x * blockDim.x + threadIdx.x; 
		if(id < wLeng) 
		{
			if(mask[id]> rate)
				mask[id]=1;
			else
				mask[id]=0;
		}
	}
}


//block<<<weightLeng/threadNum>>>
//thread<<<min(1024,Leng)>>>
__global__ void activeDropout(precision* src, precision* dest,precision* mask,float scale,unsigned int size) 
{ 
	for(int i = 0; i < size; i += blockDim.x * gridDim.x) 
	{ 
		int id = i + blockIdx.x * blockDim.x + threadIdx.x; 
		if(id < size) 
			dest[id] = src[id] * mask[id] * scale;
	}
}


precision CDropoutLayerGPU::feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	if(m_param.phase == TRAIN)
	{
		float threshold_=m_param.dropoutParam.dropoutRate;
		float scale=1;
		if(threshold_!=1)
			scale=1 / (1 - threshold_);

		CURAND_ERROR(CTools::cudaRandD(m_mask.gpuData,m_mask.size(),CTools::NORMAL));
		int size=bottoms[0]->size();
		dim3 threads2 = min(1024, size); 
		dim3 blocks2  = min(65535, (size + threads2.x - 1) / threads2.x); 
		bernoulli<<<blocks2,threads2>>>(m_mask.gpuData,m_mask.size(),1-threshold_);
		cudaError_t cudaStat=cudaDeviceSynchronize();
		CUDA_ERROR(cudaStat);

		activeDropout<<<blocks2,threads2>>>(bottoms[0]->gpuData,tops[0]->gpuData,m_mask.gpuData,scale,size);
		cudaStat=cudaDeviceSynchronize();
		CUDA_ERROR(cudaStat);
	}
	return NET_SUCCESS;
}

int CDropoutLayerGPU::backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms)
{
	if(m_param.phase == TRAIN)
	{
		float threshold_=m_param.dropoutParam.dropoutRate;
		float scale=1;
		if(threshold_!=1)
			scale=1 / (1 - threshold_);

		int size=bottoms[0]->size();
		dim3 threads2 = min(1024, size); 
		dim3 blocks2  = min(65535, (size + threads2.x - 1) / threads2.x); 
		activeDropout<<<blocks2,threads2>>>(tops[0]->gpuDiff,bottoms[0]->gpuDiff,m_mask.gpuData,scale,size);
		cudaError_t cudaStat=cudaDeviceSynchronize();
		CUDA_ERROR(cudaStat);
	}
	return NET_SUCCESS;
}