#include"LayerBaseGPU.cuh"

__global__ void setGPUWeightValue(int index,precision* weight,precision value)
{
	weight[index]=value;
}

__global__ void getGPUWeightValue(precision* value,int index,precision* weight)
{
	*value=weight[index];
}

int CLayerBaseGPU::save(FILE* fp)
{
	cudaError_t cudaStat;
	precision* pTmp=new precision[m_weightLen+m_param.curNumFeature];
	DL_ASSER(pTmp);

	cudaStat=cudaMemcpy(pTmp,m_weight.gpuData,sizeof(precision)*m_weightLen,cudaMemcpyDeviceToHost);
	CUDA_ERROR(cudaStat);

	cudaStat=cudaMemcpy(pTmp+m_weightLen,m_bias.gpuData,sizeof(precision)*m_param.curNumFeature,cudaMemcpyDeviceToHost);
	CUDA_ERROR(cudaStat);

	fwrite(pTmp,sizeof(precision)*(m_weightLen+m_param.curNumFeature),1,fp);

	cudaStat=cudaMemcpy(pTmp,m_vecWeight,sizeof(precision)*m_weightLen,cudaMemcpyDeviceToHost);
	CUDA_ERROR(cudaStat);

	cudaStat=cudaMemcpy(pTmp+m_weightLen,m_vecBias,sizeof(precision)*m_param.curNumFeature,cudaMemcpyDeviceToHost);
	CUDA_ERROR(cudaStat);

	fwrite(pTmp,sizeof(precision)*(m_weightLen+m_param.curNumFeature),1,fp);

	delete [] pTmp;
	return NET_SUCCESS;
}

int CLayerBaseGPU::load(FILE*fp)
{
	cudaError_t cudaStat;
	precision* pTmp=new precision[m_weightLen+m_param.curNumFeature];
	DL_ASSER(pTmp!=NULL);

	fread(pTmp,sizeof(precision)*(m_weightLen+m_param.curNumFeature),1,fp);
	cudaStat=cudaMemcpy(m_weight.gpuData,pTmp,sizeof(precision)*m_weightLen,cudaMemcpyHostToDevice);
	CUDA_ERROR(cudaStat);

	cudaStat=cudaMemcpy(m_bias.gpuData,pTmp+m_weightLen,sizeof(precision)*m_param.curNumFeature,cudaMemcpyHostToDevice);
	CUDA_ERROR(cudaStat);

	fread(pTmp,sizeof(precision)*(m_weightLen+m_param.curNumFeature),1,fp);
	cudaStat=cudaMemcpy(m_vecWeight,pTmp,sizeof(precision)*m_weightLen,cudaMemcpyHostToDevice);
	CUDA_ERROR(cudaStat);

	cudaStat=cudaMemcpy(m_vecBias,pTmp+m_weightLen,sizeof(precision)*m_param.curNumFeature,cudaMemcpyHostToDevice);
	CUDA_ERROR(cudaStat);

	delete [] pTmp;
	return NET_SUCCESS;
}

int CLayerBaseGPU::setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs)
{
	if(m_weightLen == 0 || m_param.curNumFeature == 0)
		return NET_SUCCESS;
	cudaError_t cudaStat;
	cudaStat=cudaMalloc((void**)&m_vecBias,sizeof(precision)*m_param.curNumFeature);
	CUDA_ERROR(cudaStat);

	cudaStat=cudaMemset(m_vecBias,0,sizeof(precision)*m_param.curNumFeature);
	CUDA_ERROR(cudaStat);

	m_bias.create(1,1,1,m_param.curNumFeature);

	cudaStat=cudaMemset(m_bias.gpuData,0,sizeof(precision)*m_param.curNumFeature);
	CUDA_ERROR(cudaStat);

	cudaStat=cudaMalloc((void**)&m_biasGrad,sizeof(precision)*m_param.curNumFeature);
	CUDA_ERROR(cudaStat);

	m_weight.create(1,1,1,m_weightLen);
#ifndef FLOAT_TYPE
	CURAND_ERROR(CTools::cudaRandD(m_weight.gpuData,m_weightLen,CTools::GAUSS));
#else
	CURAND_ERROR(CTools::cudaRandF(m_weight.gpuData,m_weightLen,CTools::GAUSS));
#endif

	cudaStat=cudaMalloc((void**)&m_weightGrad,sizeof(precision)*m_weightLen);
	CUDA_ERROR(cudaStat);

	cudaStat=cudaMalloc((void**)&m_vecWeight,sizeof(precision)*m_weightLen);
	CUDA_ERROR(cudaStat);

	cudaStat=cudaMemset(m_vecWeight,0,sizeof(precision)*m_weightLen);
	CUDA_ERROR(cudaStat);

	return CLayer::setup(inputs,outputs);
}

void CLayerBaseGPU::freeMem()
{
	GPU_FREE(m_vecBias);
	GPU_FREE(m_vecWeight);
	GPU_FREE(m_weightGrad);
	GPU_FREE(m_biasGrad);
}

int CLayerBaseGPU::updateWeight(float mom,float baseLR)
{
	if(m_weightLen == 0)
		return NET_SUCCESS;

	float biasLR=baseLR*m_param.biasLearnRatio;
	float weightsLR=baseLR*m_param.weightLearnRatio;
	int inputSize=m_weightLen/m_param.curNumFeature;
	int threadNum=min(1024,inputSize);
	g_weightAndBiasAdd<<<m_param.curNumFeature,threadNum>>>(m_weight.gpuData,m_weightGrad,
												      m_vecWeight,m_bias.gpuData,
												      m_biasGrad,m_vecBias,
												      inputSize,
												      mom,
													  weightsLR,
													  biasLR);
	cudaError_t cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	return NET_SUCCESS;
}

void CLayerBaseGPU::setWeightValue(int index,precision value)
{
	setGPUWeightValue<<<1,1>>>(index,m_weight.gpuData,value);
}

precision CLayerBaseGPU::getWeightValue(int index)
{
	precision* value=NULL;
	cudaMalloc((void**)&value,sizeof(precision));
	getGPUWeightValue<<<1,1>>>(value,index,m_weight.gpuData);
	precision result=0.0;
	cudaMemcpy(&result,value,sizeof(precision),cudaMemcpyDeviceToHost);
	cudaFree(value);
	return result;
}

__global__ void setGPUBiasValue(int index,precision* bias,precision value)
{
	bias[index]=value;
}

__global__ void getGPUBiasValue(precision*value,int index,precision* bias)
{
	*value=bias[index];
}


void CLayerBaseGPU::setBiasValue(int index,precision value)
{
	setGPUBiasValue<<<1,1>>>(index,m_bias.gpuData,value);
}

precision CLayerBaseGPU::getBiasValue(int index)
{
	precision* value=NULL;
	cudaMalloc((void**)&value,sizeof(precision));
	getGPUBiasValue<<<1,1>>>(value,index,m_bias.gpuData);
	precision result=0.0;
	cudaMemcpy(&result,value,sizeof(precision),cudaMemcpyDeviceToHost);
	cudaFree(value);
	return result;
}

//树形加法计算权值的平方和
__global__ void sumOfSquares(precision *num,unsigned int dataSize,precision* result)
{
    extern __shared__ precision shared[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

	int THREAD_NUM=blockDim.x;
	int BLOCK_NUM=gridDim.x;

    shared[tid] = 0;
    for(int i = bid * THREAD_NUM + tid; i < dataSize;
        i += BLOCK_NUM * THREAD_NUM) {
       shared[tid] += num[i] * num[i];
    }

    __syncthreads();
    int offset = THREAD_NUM / 2;
    while(offset > 0) {
        if(tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        offset >>= 1;
        __syncthreads();
    }

    if(tid == 0) {
        result[bid] = shared[0];
    }
}

precision getWeightCost(precision* devWeight,unsigned int dataSize)
{
	const int BLOCK_NUM=32;
	const int THREAD_NUM=256;

	precision* result=NULL;

	cudaError_t cudaStat=cudaMalloc((void**) &result, sizeof(precision) * BLOCK_NUM);
	CUDA_ERROR(cudaStat);

	sumOfSquares<<<BLOCK_NUM, THREAD_NUM,
        THREAD_NUM * sizeof(precision)>>>(devWeight,dataSize,result);
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	precision sum[BLOCK_NUM];
    cudaStat=cudaMemcpy(sum, result, sizeof(precision) * BLOCK_NUM,cudaMemcpyDeviceToHost);
	CUDA_ERROR(cudaStat);

    cudaStat=cudaFree(result);
	CUDA_ERROR(cudaStat);

   precision finSum = 0;
    for(int i = 0; i < BLOCK_NUM; i++)
        finSum += sum[i];
	return finSum;
}

//block<<<weightLeng/threadNum>>>
//thread<<<min(1024,weightLeng)>>>
__global__ void fullWeightGrad(precision* wgrad, precision* weight,int wLeng,float lambda, int batch) 
{ 
	for(int i = 0; i < wLeng; i += blockDim.x * gridDim.x) 
	{ 
		int id = i + blockIdx.x * blockDim.x + threadIdx.x; 
		if(id < wLeng) 
			wgrad[id] = (wgrad[id] / batch + lambda * weight[id]); 
	}
}

//block<<<numFeature>>>
//thread<<<min(512)>>>
//share<<<sizeof(precision)*threadNum>>>
__global__ void g_computeBiasGrad(precision* delta,precision* grads,int batch)
{
	extern __shared__ precision _sum[];
	int featureNo=blockIdx.x;
	int tid=threadIdx.x;

	_sum[tid]=0.0;

	for(int i=tid;i<batch;i+=blockDim.x)
	{
		int deltaIndex=i*gridDim.x+featureNo;
		_sum[tid]+=delta[deltaIndex];
	}

	int len = blockDim.x; 
	 while(len != 1) 
	 { 
		 __syncthreads(); 
		 int skip = (len + 1) >> 1; 
		 if(threadIdx.x < (len >> 1)) 
		 { 
			 _sum[threadIdx.x] +=_sum[threadIdx.x + skip]; 
		 } 
		 len = (len + 1) >> 1; 
	 } 

	 if(threadIdx.x==0)
		grads[featureNo]=_sum[0]/batch;
}

void CLayerBaseGPU::computeBiasGrad(precision* delta, precision* grads,int batch)
{
	int threadNum=512;
	g_computeBiasGrad<<<m_param.curNumFeature,threadNum,sizeof(precision)*threadNum>>>(delta,
							                                                           grads,
							                                                           batch);
	cudaError_t cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);
}

//block<<<curNumFeature>>>
//thread<<<min(maxThread,oneFeatureWeightSize)>>>
__global__ void g_weightAndBiasAdd(precision* weights,
								   precision* weightGrads,
								   precision* vec_weight,
								   precision* bias,
								   precision* biasGrad,
								   precision* vec_bias,
								   int oneFeatureWeightSize,
								   float mom,
								   float weightsLR,
								   float biasLR)
{
	int featureNo=blockIdx.x;
	for(int i=threadIdx.x;i<oneFeatureWeightSize;i+=blockDim.x)
	{
		int index=oneFeatureWeightSize*featureNo+i;
		vec_weight[index]=vec_weight[index]*mom+weightGrads[index]*weightsLR;
		weights[index]=weights[index]-vec_weight[index];
	}
	if(threadIdx.x==0)
	{
		vec_bias[featureNo]=vec_bias[featureNo]*mom+biasGrad[featureNo]*biasLR;
		bias[featureNo]-=vec_bias[featureNo];
	}
}


//block<<<batch,numFeature>>>
//thread<<<min(maxThreadNum,inputNumFeature)>>>
//share<<sizeof(precision)*threadNum>>>
__global__ void g_fullConnect(precision* srcData,
						      precision* weight,
						      int inputNumFeature,
						      precision* fullData,
						      precision* bias
						   )
{
	extern __shared__ precision featureSum[];

	int srcNo=blockIdx.x;
	int featureNo=blockIdx.y;
	int tId=threadIdx.x;
	
	int fullDataIndex=srcNo*gridDim.y+featureNo;

	featureSum[tId]=0;
	__syncthreads();

	for(int i=tId;i<inputNumFeature;i+=blockDim.x)
	{
		int dataIndex=srcNo*inputNumFeature+i;
		int weightIndex=featureNo*inputNumFeature+i;
		featureSum[tId]+=srcData[dataIndex]*weight[weightIndex];
	}
	 __syncthreads();

	 int len = blockDim.x; 
	 while(len != 1) 
	 { 
		 __syncthreads(); 
		 int skip = (len + 1) >> 1; 
		 if(threadIdx.x < (len >> 1)) 
		 { 
			 featureSum[threadIdx.x] +=featureSum[threadIdx.x + skip]; 
		 } 
		 len = (len + 1) >> 1; 
	 } 
	 __syncthreads();

	 if(threadIdx.x==0)
		 fullData[fullDataIndex]=featureSum[0]+bias[featureNo];
}