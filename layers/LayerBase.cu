#include"LayerBase.cuh"
#include<cublas_v2.h>

cublasHandle_t g_handle=NULL;

__global__ void setGPUWeightValue(int index,double* weight,double value)
{
	weight[index]=value;
}

__global__ void getGPUWeightValue(double*value,int index,double* weight)
{
	*value=weight[index];
}

int CLayer::save(FILE* fp)
{
	cudaError_t cudaStat;
	double* pTmp=new double[m_weightLen+m_curNumFeature];
	cudaStat=cudaMemcpy(pTmp,m_weight,sizeof(double)*m_weightLen,cudaMemcpyDeviceToHost);
	if(cudaStat!=cudaSuccess)
	{	
		printf ("(Save file)device memory cudaMemcpy failed\n"); 
		return -2;
	}
	cudaStat=cudaMemcpy(pTmp+m_weightLen,m_bias,sizeof(double)*m_curNumFeature,cudaMemcpyDeviceToHost);
	if(cudaStat!=cudaSuccess)
	{	
		printf ("(Save file)device memory cudaMemcpy failed\n"); 
		return -2;
	}
	fwrite(pTmp,sizeof(double)*(m_weightLen+m_curNumFeature),1,fp);
	delete [] pTmp;
	return PK_SUCCESS;
}

CLayer::CLayer()
{
	m_vecBias=m_vecWeight=m_weight=m_bias=m_weightGrad=m_biasGrad=m_delta=NULL;
}

int CLayer::load(FILE*fp)
{
	cudaError_t cudaStat;
	double* pTmp=new double[m_weightLen+m_curNumFeature];
	fread(pTmp,sizeof(double)*(m_weightLen+m_curNumFeature),1,fp);
	cudaStat=cudaMemcpy(m_weight,pTmp,sizeof(double)*m_weightLen,cudaMemcpyHostToDevice);
	if(cudaStat!=cudaSuccess)
	{	
		printf ("(Load file)device memory cudaMemcpy failed\n"); 
		return -2;
	}
	cudaStat=cudaMemcpy(m_bias,pTmp+m_weightLen,sizeof(double)*m_curNumFeature,cudaMemcpyHostToDevice);
	if(cudaStat!=cudaSuccess)
	{	
		printf ("(Load file)device memory cudaMemcpy failed\n"); 
		return -2;
	}
	delete [] pTmp;
	return PK_SUCCESS;
}

int CLayer::initMem()
{
	cudaError_t cudaStat;
	cudaStat=cudaMalloc((void**)&m_vecBias,sizeof(double)*m_curNumFeature);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMalloc failed\n"); 
		freeMem();
		return -1;
	}
	cudaStat=cudaMemset(m_vecBias,0,sizeof(double)*m_curNumFeature);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMemset failed\n"); 
		freeMem();
		return -2;
	}

	cudaStat=cudaMalloc((void**)&m_bias,sizeof(double)*m_curNumFeature);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMalloc failed\n"); 
		freeMem();
		return -1;
	}
	cudaStat=cudaMemset(m_bias,0,sizeof(double)*m_curNumFeature);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMemset failed\n"); 
		freeMem();
		return -2;
	}

	cudaStat=cudaMalloc((void**)&m_biasGrad,sizeof(double)*m_curNumFeature);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMalloc failed\n"); 
		freeMem();
		return -1;
	}

	cudaStat=cudaMalloc((void**)&m_weight,sizeof(double)*m_weightLen);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMalloc failed\n");  
		freeMem();
		return -1;
	}
	double* pData=new double[m_weightLen];
	randn(pData,m_weightLen);
	cudaStat=cudaMemcpy(m_weight,pData,sizeof(double)*m_weightLen, cudaMemcpyHostToDevice);
	delete [] pData;
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMemcpy failed\n"); 
		freeMem();
		return -1;
	}

	cudaStat=cudaMalloc((void**)&m_weightGrad,sizeof(double)*m_weightLen);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMalloc failed\n");  
		freeMem();
		return -1;
	}

	cudaStat=cudaMalloc((void**)&m_vecWeight,sizeof(double)*m_weightLen);
	if(cudaStat!=cudaSuccess)
	{
		printf("device memory cudaMalloc failed\n");  
		freeMem();
		return -1;
	}

	cudaStat=cudaMemset(m_vecWeight,0,sizeof(double)*m_weightLen);
	if(cudaStat!=cudaSuccess)
	{
		printf("device memory cudaMemset failed\n");  
		freeMem();
		return -2;
	}
	return PK_SUCCESS;
}

void CLayer::freeMem()
{
	GPU_FREE(m_vecBias);
	GPU_FREE(m_vecWeight);
	GPU_FREE(m_weight);
	GPU_FREE(m_bias);
	GPU_FREE(m_weightGrad);
	GPU_FREE(m_biasGrad);
	GPU_FREE(m_delta);
}

void CLayer::setWeightValue(int index,double value)
{
	setGPUWeightValue<<<1,1>>>(index,m_weight,value);
}

double CLayer::getWeightValue(int index)
{
	double*value=NULL;
	cudaMalloc((void**)&value,sizeof(double));
	getGPUWeightValue<<<1,1>>>(value,index,m_weight);
	double result=0.0;
	cudaMemcpy(&result,value,sizeof(double),cudaMemcpyDeviceToHost);
	cudaFree(value);
	return result;
}

__global__ void setGPUBiasValue(int index,double* bias,double value)
{
	bias[index]=value;
}

__global__ void getGPUBiasValue(double*value,int index,double* bias)
{
	*value=bias[index];
}


void CLayer::setBiasValue(int index,double value)
{
	setGPUBiasValue<<<1,1>>>(index,m_bias,value);
}

double CLayer::getBiasValue(int index)
{
	double*value=NULL;
	cudaMalloc((void**)&value,sizeof(double));
	getGPUBiasValue<<<1,1>>>(value,index,m_bias);
	double result=0.0;
	cudaMemcpy(&result,value,sizeof(double),cudaMemcpyDeviceToHost);
	cudaFree(value);
	return result;
}

//树形加法计算权值的平方和
__global__ void sumOfSquares(double *num,unsigned int dataSize, double* result)
{
    extern __shared__ double shared[];
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

double getWeightCost(double* devWeight,unsigned int dataSize)
{
	const int BLOCK_NUM=32;
	const int THREAD_NUM=256;

	double* result=NULL;

	cudaMalloc((void**) &result, sizeof(double) * BLOCK_NUM);

	sumOfSquares<<<BLOCK_NUM, THREAD_NUM,
        THREAD_NUM * sizeof(double)>>>(devWeight,dataSize,result);

	cudaDeviceSynchronize();

	double sum[BLOCK_NUM];
    cudaMemcpy(sum, result, sizeof(double) * BLOCK_NUM,cudaMemcpyDeviceToHost);

    cudaFree(result);

   double finSum = 0;
    for(int i = 0; i < BLOCK_NUM; i++)
        finSum += sum[i];
	return finSum;
}

//block<<<weightLeng/threadNum>>>
//thread<<<min(1024,weightLeng)>>>
__global__ void fullWeightGrad2(double* wgrad, double* weight,float* dropW,int wLeng, double lambda, int batch) 
{ 
	for(int i = 0; i < wLeng; i += blockDim.x * gridDim.x) 
	{ 
		int id = i + blockIdx.x * blockDim.x + threadIdx.x; 
		if(id < wLeng) 
			wgrad[id] = (wgrad[id] / batch + lambda * weight[id])*dropW[id]; 
	}
}

//block<<<numFeature,inputNumFeature>>>
//thread<<<1>>>
__global__ void fullWeightGrad(double* wgrad, double* weight, double lambda, int batch) 
{ 
	int id=blockIdx.x*gridDim.y+blockIdx.y;
	wgrad[id] = wgrad[id] / batch + lambda * weight[id]; 
}

//block<<<numFeature>>>
//thread<<<min(1024,batch)>>>
//share<<<sizeof(double)*threadNum>>>
__global__ void fullBiasGrad( double* delta, double* grads,int batch)
{
	extern __shared__ double _sum[];
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

//block<<<curNumFeature>>>
//thread<<<min(maxThread,oneFeatureWeightSize)>>>
__global__ void g_weightAndBiasAdd(double* weights,
								 double* weightGrads,
								 double* vec_weight,
								 double* bias,
								 double* biasGrad,
								 double* vec_bias,
								 int oneFeatureWeightSize,
								 float mom,
								 float alpha)
{
	int featureNo=blockIdx.x;
	for(int i=threadIdx.x;i<oneFeatureWeightSize;i+=blockDim.x)
	{
		int index=oneFeatureWeightSize*featureNo+i;
		vec_weight[index]=vec_weight[index]*mom+weightGrads[index]*alpha;
		weights[index]=weights[index]-vec_weight[index];
	}
	if(threadIdx.x==0)
	{
		vec_bias[featureNo]=vec_bias[featureNo]*mom+biasGrad[featureNo]*alpha;
		bias[featureNo]-=vec_bias[featureNo];
	}
}


//block<<<batch,numFeature>>>
//thread<<<min(maxThreadNum,inputNumFeature)>>>
//share<<sizeof(double)*threadNum>>>
__global__ void g_fullConnect(
						   double* srcData,
						   double* weight,
						   int inputNumFeature,
						   double* fullData,
						   double* bias
						   )
{
	extern __shared__ double featureSum[];

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



void matrixMulTB(double * x,int colsX,double*y,int rowsY,int colsY,double*z,int colsZ) 
{  
	cublasStatus_t ret=CUBLAS_STATUS_SUCCESS;
	if(g_handle==NULL)
		ret = cublasCreate(&g_handle);
	if(ret != CUBLAS_STATUS_SUCCESS)
		printf( "cublasSgemm returned error code");

 	cublasStatus_t stat; 
 	double alpha = 1.0; 
 	double beta = 0.0; 
 	stat = cublasDgemm( 
 		g_handle,  
 		CUBLAS_OP_N, 
 		CUBLAS_OP_T, 
 		colsY, 
		colsX, 
 		rowsY, 
 		&alpha, 
 		y, 
 		colsY, 
 		x, 
 		colsX, 
 		&beta, 
 		z, 
 		colsZ); 
 	cudaDeviceSynchronize(); 
 	if(stat != CUBLAS_STATUS_SUCCESS) 
	{ 
 		printf("matrixMulTA cublasSgemm error\n"); 
		exit(0); 
	} 
} 

void matrixMul(double * x,int rowsX,int colsX,double*y,int rowsY,int colsY,double*z,int colsZ) 
{  
	cublasStatus_t ret=CUBLAS_STATUS_SUCCESS;
	if(g_handle==NULL)
		ret = cublasCreate(&g_handle);
	if(ret != CUBLAS_STATUS_SUCCESS)
		printf( "cublasSgemm returned error code");

 	cublasStatus_t stat; 
 	double alpha = 1.0; 
 	double beta = 0.0; 
 	stat = cublasDgemm( 
 		g_handle,  
 		CUBLAS_OP_N, 
 		CUBLAS_OP_N, 
 		colsY, 
		rowsX, 
 		rowsY, 
 		&alpha, 
 		y, 
 		colsY, 
 		x, 
 		colsX, 
 		&beta, 
 		z, 
 		colsZ); 
 	cudaDeviceSynchronize(); 
 	if(stat != CUBLAS_STATUS_SUCCESS) 
	{ 
 		printf("matrixMulTA cublasSgemm error\n"); 
		exit(0); 
	} 
} 

void matrixMulTA(double * x,int rowsX,int colsX,double*y,int rowsY,int colsY,double*z,int colsZ) 
{  
	cublasStatus_t ret=CUBLAS_STATUS_SUCCESS;
	if(g_handle==NULL)
		ret = cublasCreate(&g_handle);
	if(ret != CUBLAS_STATUS_SUCCESS)
		printf( "cublasSgemm returned error code");

 	cublasStatus_t stat; 
 	double alpha = 1.0; 
 	double beta = 0.0; 
 	stat = cublasDgemm( 
		g_handle,  
 		CUBLAS_OP_T, 
 		CUBLAS_OP_N, 
 		rowsY, 
 		rowsX, 
 		colsY, 
 		&alpha, 
 		y, 
 		colsY, 
 		x, 
 		colsX, 
 		&beta, 
 		z, 
 		colsZ); 
 	cudaDeviceSynchronize(); 
 	if(stat != CUBLAS_STATUS_SUCCESS) 
	{ 
 		printf("matrixMulTA cublasSgemm error\n"); 
		exit(0); 
	} 
} 