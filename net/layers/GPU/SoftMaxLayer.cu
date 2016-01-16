#include"softmaxLayer.cuh"

int CSoftMaxLayerGPU::setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs)
{
	int inputSize=inputs[0]->size()/inputs[0]->num;;
	m_weightLen=m_param.curNumFeature*inputSize;

	outputs[0]->create(inputs[0]->num,m_param.curNumFeature,1,1);

	return CLayerBaseGPU::setup(inputs,outputs);
}

//block<<<batch>>>
//thread<<<min(1024,numFeature)>>>
__global__ void feedForwardAddBias(precision* activeData,int numFeature,precision* bias)
{
	int index=blockIdx.x*numFeature;
	for(int id = 0; id < numFeature; id += blockDim.x) 
	{ 
		int idx = id + threadIdx.x; 
		if(idx < numFeature)
			activeData[index+idx]+=bias[idx];
	}
}

//block<<<1>>>
//thread<<<min(512,batch)>>>
__global__ void getSoftMaxCost(precision* sum,precision* softmaxOutput,precision* labels,int curNumFeature,int batch)
{
	extern __shared__ precision shared[];
	int tid=threadIdx.x;
	shared[tid]=0;
	for(int id = tid; id < batch; id += blockDim.x) 
	{ 
		int idx = id*curNumFeature+labels[id];
		shared[tid]+=log(softmaxOutput[idx]);
	}
	
	__syncthreads();
    int offset = blockDim.x / 2;
    while(offset > 0) {
        if(tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        offset >>= 1;
        __syncthreads();
    }

    if(tid == 0) {
        sum[0] = shared[0];
    }
}

//block<<<num>>>
//thread<<<512>>>
__global__ void softMaxPredict(precision* predicData,precision* inputData,int curNumFeature)
{
	extern __shared__ precision shared[];
	int tid=threadIdx.x;
	shared[tid]=-999999;
	for(int id = tid; id < curNumFeature; id += blockDim.x) 
	{ 
		int idx = blockIdx.x*curNumFeature+id;
		if(shared[tid]<inputData[idx])
			shared[tid]=inputData[idx];
	}

	__syncthreads();
    int offset = blockDim.x / 2;
    while(offset > 0) {
        if(tid < offset) 
		{
			if(shared[tid]<shared[tid + offset])
				shared[tid]=shared[tid + offset];
        }
        offset >>= 1;
        __syncthreads();
    }

	precision maxValue=shared[0];
	 __syncthreads();

	shared[tid]=0;
	for(int id = tid; id < curNumFeature; id += blockDim.x) 
	{ 
		int idx = blockIdx.x*curNumFeature+id;
		predicData[idx]=exp(inputData[idx]-maxValue);
		shared[tid]+=predicData[idx];
	}

	__syncthreads();
    offset = blockDim.x / 2;
    while(offset > 0) 
	{
        if(tid < offset)
			shared[tid] += shared[tid + offset];
        offset >>= 1;
        __syncthreads();
    }

	for(int id = tid; id < curNumFeature; id += blockDim.x) 
	{ 
		int idx = blockIdx.x*curNumFeature+id;
		predicData[idx]/=shared[0];
	}

}

precision CSoftMaxLayerGPU::feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	int inputSize=bottoms[0]->size()/bottoms[0]->num;
#ifndef FLOAT_TYPE
	CTools::cudaMatrixMulD(bottoms[0]->gpuData,bottoms[0]->num,inputSize,m_weight.gpuData,m_param.curNumFeature,inputSize,tops[0]->gpuData,m_param.curNumFeature,CTools::TRANSPOSE_Y);
#endif

	cudaError_t cudaStat;
	dim3 blocks=bottoms[0]->num;
	dim3 threads=min(1024,m_param.curNumFeature);
	feedForwardAddBias<<<blocks,threads>>>(tops[0]->gpuData,m_param.curNumFeature,m_bias.gpuData);
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	int threadNum=512;
	softMaxPredict<<<blocks,threadNum,sizeof(precision)*threadNum>>>(tops[0]->gpuData,tops[0]->gpuData,m_param.curNumFeature);
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	int dataSize=m_param.curNumFeature*inputSize;
	precision finSum=getWeightCost(m_weight.gpuData,dataSize);
	precision* gpuPredSum;
	precision  cpuPredSum;

	cudaMalloc((void**)&gpuPredSum,sizeof(precision));

	threadNum=min(512,tops[0]->num);
	getSoftMaxCost<<<1,threadNum,sizeof(precision)*threadNum>>>(gpuPredSum,tops[0]->gpuData,bottoms[1]->gpuData,m_param.curNumFeature,tops[0]->num);
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	cudaStat=cudaMemcpy(&cpuPredSum,gpuPredSum,sizeof(precision),cudaMemcpyDeviceToHost);
	CUDA_ERROR(cudaStat);
	cudaFree(gpuPredSum);

	cpuPredSum/=-bottoms[0]->num;
	return cpuPredSum+finSum*m_param.lambda/2;
}

//block<<<1>>>
//thread<<<min(512,batch)>>>
__global__ void setDiffFromLabel(precision* softmaxOutputDiff,precision* labels,int curNumFeature,int batch)
{
	int tid=threadIdx.x;
	for(int id = tid; id < batch; id += blockDim.x) 
	{ 
		int idx = id*curNumFeature+labels[id];
		softmaxOutputDiff[idx]=1;
	}
}

//block<<<batch,m_curNumFeature>>>
//thread<<<1>>>
__global__ void softmaxGetTruthDelta(precision* truthData,precision* probData)
{
	int imgNo=blockIdx.x;
	int featureNo=blockIdx.y;
	int index=imgNo*gridDim.y+featureNo;
	truthData[index]=-(truthData[index]-probData[index]);
}

//block<<<batch,m_inputNumFeature>>>
//thread<<<m_curNumFeature>>>
//share<<<sizeof(precision)*m_curNumFeature>>>
__global__ void softmaxGetDelta(precision* truthDelta,
								precision* weight,
								int curNumFeature,
								precision* delta
								)
{

	extern __shared__ precision shared[];

	int imgNo=blockIdx.x;
	int inputFeatureNo=blockIdx.y;
	int featureNo=threadIdx.x;
	int tIndex=imgNo*curNumFeature+featureNo;
	int wIndex=featureNo*gridDim.y+inputFeatureNo;
	int dIndex=imgNo*gridDim.y+inputFeatureNo;
	shared[featureNo]=weight[wIndex]*truthDelta[tIndex];
	if(featureNo==0)
	{
		for(int i=0;i<curNumFeature;i++)
			delta[dIndex]+=shared[i];
	}
}

int CSoftMaxLayerGPU::backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms)
{
	cudaError_t cudaStat;
	cudaStat=cudaMemset(tops[0]->gpuDiff,0,sizeof(precision)*tops[0]->size());
	CUDA_ERROR(cudaStat);

	dim3 threads=min(512,tops[0]->num);
	setDiffFromLabel<<<1,threads>>>(tops[0]->gpuDiff,bottoms[1]->gpuData,m_param.curNumFeature,tops[0]->num);
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	dim3 blocks(tops[0]->num,m_param.curNumFeature);
	softmaxGetTruthDelta<<<blocks,1>>>(tops[0]->gpuDiff,tops[0]->gpuData);
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	int inputSize=bottoms[0]->size()/bottoms[0]->num;
#ifndef FLOAT_TYPE
	CTools::cudaMatrixMulD(tops[0]->gpuDiff,tops[0]->num,m_param.curNumFeature,m_weight.gpuData,m_param.curNumFeature,inputSize,bottoms[0]->gpuDiff,inputSize,CTools::NORMAL_XY);
#endif

	return getGrad(tops,bottoms);
}

int CSoftMaxLayerGPU::getGrad(std::vector<Blob<precision>*>& tops,std::vector<Blob<precision>*>& bottoms)
{
	cudaError_t cudaStat;

	int inputSize=bottoms[0]->size()/bottoms[0]->num;
#ifndef FLOAT_TYPE
	CTools::cudaMatrixMulD(tops[0]->gpuDiff,bottoms[0]->num,m_param.curNumFeature,bottoms[0]->gpuData,bottoms[0]->num,inputSize,m_weightGrad,inputSize,CTools::TRANSPOSE_X);
#endif

	dim3 threads2 = min(1024,m_weightLen); 
	dim3 blocks2  = min(1024, (m_weightLen + threads2.x - 1) / threads2.x); 

	fullWeightGrad<<<blocks2,threads2>>>(m_weightGrad,m_weight.gpuData,m_weightLen,m_param.lambda,bottoms[0]->num);
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	this->computeBiasGrad(tops[0]->gpuDiff,m_biasGrad,bottoms[0]->num);
	return NET_SUCCESS;
}