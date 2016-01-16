#include"Accuracy.cuh"

int CAccuracyGPU::setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs)
{
	outputs[0]->create(1,1,1,1);
	m_corrects.create(inputs[0]->num,1,1,1);
	return 0;
}

//block<<<<1>>>
//thread<<<batch>>>
__global__ void getCorrects(precision* corrects,precision* inputData,precision* labels,int dataChannel)
{
	int tid=threadIdx.x;
	precision max=0.0;
	int maxIndex=0;
	for(int i=0;i<dataChannel;i++)
	{
		int index=tid*dataChannel+i;
		if(max<inputData[index])
		{
			max=inputData[index];
			maxIndex=i;
		}
	}
	if(maxIndex==(int)labels[tid])
		corrects[tid]=1;
	else
		corrects[tid]=0;
}

//block<<<1>>>
//thread<<<min(512,batch)>>>
__global__ void getCorrectSum(precision* sum,precision* correct,int batch)
{
	extern __shared__ precision shared[];
	int tid=threadIdx.x;
	shared[tid]=0;

	for(int id = tid; id < batch; id += blockDim.x) 
		shared[tid]+=correct[id];
	
	__syncthreads();

    int offset = blockDim.x / 2;
    while(offset > 0) {
        if(tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        offset >>= 1;
        __syncthreads();
    }

    if(tid == 0)
        sum[0] += shared[0];
}


precision CAccuracyGPU::feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	cudaError_t cudaStat=cudaSuccess;
	getCorrects<<<1,bottoms[1]->num>>>(m_corrects.gpuData,
									   bottoms[0]->gpuData,
									   bottoms[1]->gpuData,
									   bottoms[0]->dataChannel
									   );
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	int threadNum=512;//保证是2的整数倍，因为求和是树形加法
	getCorrectSum<<<1,threadNum,sizeof(precision)*threadNum>>>(tops[0]->gpuData,
															   m_corrects.gpuData,
															   bottoms[1]->num
															   );
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	return NET_SUCCESS;
}