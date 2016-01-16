#include"relulayer.cuh"

//block<<<weightLeng/threadNum>>>
//thread<<<min(1024,Leng)>>>
__global__ void activeRelu(precision* src, precision* dest,unsigned int size) 
{ 
	for(int i = 0; i < size; i += blockDim.x * gridDim.x) 
	{ 
		int id = i + blockIdx.x * blockDim.x + threadIdx.x; 
		if(id < size) 
		{
			dest[id] = ::max(0.0,src[id]);
		}
	}
}


precision CReluLayerGPU::feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	int size=bottoms[0]->size();
	dim3 threads2 = min(1024, size); 
	dim3 blocks2  = min(65535,(size + threads2.x - 1) / threads2.x); 
	activeRelu<<<blocks2,threads2>>>(bottoms[0]->gpuData,tops[0]->gpuData,size);
	cudaError_t cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);
	return 0;
}

//block<<<weightLeng/threadNum>>>
//thread<<<min(1024,Leng)>>>
__global__ void d_activeRelu(precision* src, precision* dest,unsigned int size) 
{ 
	for(int i = 0; i < size; i += blockDim.x * gridDim.x) 
	{ 
		int id = i + blockIdx.x * blockDim.x + threadIdx.x; 
		if(id < size) 
		{
			if(src[id]>0.0)
				dest[id] = 1.0;
			else
			    dest[id] = 0.0;
		}
	}
}

int CReluLayerGPU::backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms)
{
	int size=bottoms[0]->size();
	dim3 threads2 = min(1024, size); 
	dim3 blocks2  = min(65535, (size + threads2.x - 1) / threads2.x); 
	d_activeRelu<<<blocks2,threads2>>>(tops[0]->gpuData,bottoms[0]->gpuData,size);
	cudaError_t cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);
	return NET_SUCCESS;
}