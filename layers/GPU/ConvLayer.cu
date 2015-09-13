#include"ConvLayer.cuh"

__device__ double activeFun(double src,int type)
{
	if(type==NL_RELU)
	{
		if(src<0.0)
			return 0.0;
		else
			return src;
	}
	else if(type==NL_SOFT_PLUS)
		return ::log(1+::exp(src));
	return 1/(1+::exp(-src));
}

__device__ double d_activeFun(double src,int type)
{
	if(type==NL_RELU)
	{
		if(src>0.0)
			return 1.0;
		else
			return 0.0;

	}
	else if(type==NL_SOFT_PLUS)
		return 1/(1+::exp(-src));
	return src*(1-src);
}

//blocks<<<batch,NumFeature>>>
//thread<<<min(1024,convDim*convDim)>>>
__global__ void g_dConvActive(
						  double* deltaData,
						  double* convNoActiveData,
						  int convDim,
						  int type
						  )
{
	int convDim2=convDim*convDim;
	int index=blockIdx.x*convDim2*gridDim.y+blockIdx.y*convDim2;
	for(int idx=threadIdx.x;idx<convDim2;idx+=blockDim.x)
		deltaData[index+idx]*=d_activeFun(convNoActiveData[index+idx],type);
}


//block<<<batch,m_curNumFeature>>>
//threads<<<min(1024,convDim*convDim)>>>
__global__ void feedForwardConvolution(
							double* srcData,
							int srcDim,
							int inputNumFeature,
							double* maskData,
							int maskDim,
							int numFeature,
							double* destData,
							double* destNoactiveData,
							double* pB,
							int type
							)
{
	int srcNo=blockIdx.x;
	int featureNo=blockIdx.y;
	int conDim=srcDim-maskDim+1;
	int conDim2=conDim*conDim;
	int srcDim2=srcDim*srcDim;
	int maskDim2=maskDim*maskDim;
	
	double* pSrc=srcData+srcNo*srcDim2*inputNumFeature;
	double* pMask=maskData+featureNo*maskDim2*inputNumFeature;
	double* pDest=destData+(srcNo*numFeature+featureNo)*conDim2;
	double* pDestNoAvtive=destNoactiveData+(srcNo*numFeature+featureNo)*conDim2;

	for(int idx=threadIdx.x;idx<conDim2;idx+=blockDim.x)
	{
		int convX=idx%conDim;
		int convY=idx/conDim;
		double res=0.0;
		for(int k=0;k<inputNumFeature;k++)
		{
			double* pX1=pSrc+k*srcDim2;
			double* pX2=pMask+k*maskDim2;

			for(int x1=0;x1<maskDim;x1++)
			{
				for(int x2=0;x2<maskDim;x2++)			
				res+= pX1[(convY+x1)*srcDim+convX+x2] * pX2[x1*maskDim+x2];
			}
		}
		res+=pB[featureNo];
		pDestNoAvtive[convY*conDim+convX]=res;
		pDest[convY*conDim+convX]=activeFun(res,type);
	}
}

//block<<<m_curNumFeature,m_inputNumFeature>>>
//threads<<<min(1024,poolDim*poolDim)>>>
__global__ void normal(
						double* normalData,
						double* tmpPoolData,
						int poolDim
						)
{
	int featureNo=blockIdx.x;
	int channelNo=blockIdx.y;
	int kernelSize=gridDim.x;
	int channelSize=gridDim.y;
	int poolDim2=poolDim*poolDim;
	double* pNormal=normalData+(featureNo*channelSize+channelNo)*poolDim2;

	for(int idx=threadIdx.x;idx<poolDim2;idx+=blockDim.x)
	{
		double sum=0.0;
		for(int j=max(0,(featureNo-5)/2);j<min(kernelSize,(featureNo+5)/2);j++)
		{
			double* pTmpData= tmpPoolData+(j*channelSize+channelNo)*poolDim2;
			sum+=pTmpData[idx]*pTmpData[idx];
		}
		sum=3+sum*1e-3;
		sum=pow(sum,3);
		sum=sqrt(sqrt(sum));
		pNormal[idx]/=sum;
	}
}

void CConvLayerGPU::feedforward(double* srcData,DLparam& params)
{
	int activeType=params.activeType;
	dim3 blocks(batch,m_curNumFeature);
	dim3 threads(min(1024,m_convDim*m_convDim));
	feedForwardConvolution<<<blocks,threads>>>
							(srcData,
							 m_convDim+m_kernelDim-1,
							 m_inputNumFeature,
							 m_weight,
							 m_kernelDim,
							 m_curNumFeature,
							 m_convData,
							 m_convNoActiveData,
							 m_bias,
							 activeType);
	cudaError_t cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);
}

double CConvLayerGPU::getCost(DLparam& params)
{
	int dataSize=m_curNumFeature*m_inputNumFeature*m_kernelDim*m_kernelDim;
	double finSum=getWeightCost(m_weight,dataSize);
	return finSum*m_lambda/2;
}

//blocks<<<batch,inputNumFeature>>>
//threads<<<min(1024,curPoolDeltaDim*curPoolDeltaDim)>>>
__global__ void backDeltaFromConvToPool(
										double* convDelta,
										int convDim,
										int convNumFeature,
										double* convWeight,
										int maskDim,
										double* poolDelta,
										int poolNumFeature
										)
{
	int srcNo=blockIdx.x;
	int featureNo=blockIdx.y;

	int convDim2=convDim*convDim;
	int maskDim2=maskDim*maskDim;

	int dstSize = convDim + maskDim - 1;
	int dstSize2=dstSize*dstSize;
	int poolDim2=dstSize*dstSize;
	int edgeSize = maskDim - 1;  
	int kernel_i=0,src_i=0,kernel_j=0,src_j=0;
	
	double* pConvDetal=convDelta+srcNo*convNumFeature*convDim2;
	double* pPool=poolDelta+(srcNo*poolNumFeature+featureNo)*poolDim2;

	for(int idx=threadIdx.x;idx<dstSize2;idx+=blockDim.x)
	{
		int j=idx%dstSize;
		int i=idx/dstSize;  
		double res=0.0;
		for(int g=0;g<convNumFeature;g++)
		{
			double* pSrc=pConvDetal+g*convDim2;
			double* pMask=convWeight+g*maskDim2*poolNumFeature+featureNo*maskDim2;  
			double sum = 0.0; 

			kernel_i = maskDim - 1 - PK_MAX(0, edgeSize - i);  
			src_i = PK_MAX(0, i - edgeSize);  
			for (; kernel_i >= 0 && src_i < convDim; kernel_i--, src_i++)
			{  
				kernel_j = maskDim - 1 - PK_MAX(0, edgeSize - j);  
				src_j =PK_MAX(0, j - edgeSize);  
				for (; kernel_j >= 0 && src_j < convDim; kernel_j--, src_j++)
					sum += pSrc[src_i*convDim+src_j] * pMask[kernel_i*maskDim+kernel_j];  
			}            
			res+= sum; 		 
		} 
		pPool[i*dstSize+j]=res;
	}
}


void CConvLayerGPU::backpropagation(double*preDelta,DLparam& params)
{
	int activeType=params.activeType;
	dim3 blocks(batch,m_curNumFeature);
	dim3 threads(min(1024,m_convDim*m_convDim));
	g_dConvActive<<<blocks,threads>>>(m_delta,
									  m_convNoActiveData,
									  m_convDim,
									  activeType
									  );
	cudaError_t cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	if(preDelta!=NULL)
	{
		dim3 blocks(batch,m_inputNumFeature);
		int prePoolDim=m_convDim+m_kernelDim-1;
		threads=min(1024,prePoolDim*prePoolDim);

		backDeltaFromConvToPool<<<blocks,threads>>>(m_delta,
													m_convDim,
													m_curNumFeature,
													m_weight,
											        m_kernelDim,
											        preDelta,
											        m_inputNumFeature);
		 cudaStat=cudaDeviceSynchronize();
		 CUDA_ERROR(cudaStat);
	}
}

//blocks<<<batch,m_curNumFeature>>>
//threads<<<WeightDim,WeightDim>>>
__global__ void gradWeight(double* srcData,
							int srcDim,
							int inputNumFeature,
							double* deltaData,
							int deltaDim,
							int numFeature,
							double* weightGradData)
{
	int srcNo=blockIdx.x;
	int featureNo=blockIdx.y;
	int conDim=srcDim-deltaDim+1;
	int conDim2=conDim*conDim;
	int srcDim2=srcDim*srcDim;
	int deltaDim2=deltaDim*deltaDim;
	
	double* pSrc=srcData+srcNo*srcDim2*inputNumFeature;
	double* pMask=deltaData+srcNo*deltaDim2*gridDim.y;
	double* pGrad=weightGradData+srcNo*numFeature*inputNumFeature*conDim2+featureNo*inputNumFeature*conDim2;

	int convX=threadIdx.x;
	int convY=threadIdx.y;
	
	for(int k=0;k<inputNumFeature;k++)
	{
		double* pX1=pSrc+k*srcDim2;
		double* pX2=pMask+featureNo*deltaDim2;
		double res=0.0;

		for(int x1=0;x1<deltaDim;x1++)
		{
			for(int x2=0;x2<deltaDim;x2++)			
			{
				res+= pX1[(convY+x1)*srcDim+convX+x2] * pX2[x1*deltaDim+x2];
			}
		}
		pGrad[k*conDim2+convY*conDim+convX]=res;
	}
	
}


//blocks<<<batch,m_curNumFeature,inputNumFeature>>>
//threads<<<WeightDim,WeightDim>>>
__global__ void gradWeight2(double* srcData,
							int srcDim,
							int inputNumFeature,
							double* deltaData,
							int deltaDim,
							int numFeature,
							double* weightGradData)
{
	int srcNo=blockIdx.x;
	int featureNo=blockIdx.y;
	int conDim=srcDim-deltaDim+1;
	int conDim2=conDim*conDim;
	int srcDim2=srcDim*srcDim;
	int deltaDim2=deltaDim*deltaDim;
	
	double* pSrc=srcData+srcNo*srcDim2*inputNumFeature;
	double* pMask=deltaData+srcNo*deltaDim2*gridDim.y;
	double* pGrad=weightGradData+srcNo*numFeature*inputNumFeature*conDim2+featureNo*inputNumFeature*conDim2;

	int convX=threadIdx.x;
	int convY=threadIdx.y;
	
	int k=blockIdx.z;
//	for(int k=0;k<inputNumFeature;k++)
//	{
		double* pX1=pSrc+k*srcDim2;
		double* pX2=pMask+featureNo*deltaDim2;
		double res=0.0;

		for(int x1=0;x1<deltaDim;x1++)
		{
			for(int x2=0;x2<deltaDim;x2++)			
			{
				res+= pX1[(convY+x1)*srcDim+convX+x2] * pX2[x1*deltaDim+x2];
			}
		}
		pGrad[k*conDim2+convY*conDim+convX]=res;
//	}
}

//blocks<<<curNumFeature,inputNumFeature>>>
//thread<<<weightDim,weightDim>>>
__global__ void gradWeightAdd(
							double* weightGrad,
							double* weightTmp,
							double* weight,
							int  weightDim,
							double lambda,
							int batch)
{
	int featureNo=blockIdx.x;
	int inputFeatureNo=blockIdx.y;
	int tidX=threadIdx.x;
	int tidY=threadIdx.y;

	int weightDim2=weightDim*weightDim;

	int index=(featureNo*gridDim.y+inputFeatureNo)*weightDim2+tidY*weightDim+tidX;
	double tmp=0.0;
	
	int oneDataSize=gridDim.x*gridDim.y*weightDim2;
	for(int i=0;i<batch;i++)
		tmp+=weightTmp[index+i*oneDataSize];
	weightGrad[index]=tmp/batch+weight[index]*lambda;
}

//blocks<<<curNumFeature,batch>>>
//threads<<min(threadNum,deltaDim*deltaDim)>>>
//share<<threads*sizeof(double>>>
__global__ void addDeltaSum(double* delta,
							double* deltaSum,
							int deltaDim)
{
	extern __shared__ double _sum[];
	int featureNo=blockIdx.x;
	int imgNo=blockIdx.y;
	int tid=threadIdx.x;
	int deltaDim2=deltaDim*deltaDim;

	double* pDelta=delta+imgNo*gridDim.x*deltaDim2+featureNo*deltaDim2;
	_sum[tid] = 0.0; 
	for(int i=tid;i<deltaDim2;i+=blockDim.x)
	{
		_sum[tid]+=pDelta[i];
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
	 __syncthreads();

	if(threadIdx.x == 0) 
		deltaSum[imgNo*gridDim.x+featureNo]=_sum[0];
}

//block3<<<1>>>
//threads<<curNumFeature>>>
__global__ void gradBias(
						 double* biasGrad,
						 double* deltaSum,
						 int batch
						 )
{
	int featureNo=threadIdx.x;
	double sum=0.0;
	for(int i=0;i<batch;i++)
		sum+=deltaSum[i*blockDim.x+featureNo];
	biasGrad[featureNo]=sum/batch;
}


void CConvLayerGPU::getGrad(double* srcData)
{
	cudaError_t cudaStat=cudaSuccess;
	dim3 blocks(batch,m_curNumFeature,m_inputNumFeature);
	dim3 threads(m_kernelDim,m_kernelDim);
	int srcDim=m_convDim+m_kernelDim-1;

	gradWeight2<<<blocks,threads>>>(srcData,
								   srcDim,
							       m_inputNumFeature,
							       m_delta,
							       m_convDim,
							       m_curNumFeature,
							       m_weightTmp
								   );
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	dim3 blocks2(m_curNumFeature,m_inputNumFeature);
	dim3 threads2(m_kernelDim,m_kernelDim);

	gradWeightAdd<<<blocks2,threads2>>>(
								m_weightGrad,
							    m_weightTmp,
							    m_weight,
								m_kernelDim,
							    m_lambda,
							    batch);
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	dim3 block3(m_curNumFeature,batch);
	int threadNum=min(256,m_convDim*m_convDim);

	addDeltaSum<<<block3,threadNum,sizeof(double)*threadNum>>>(m_delta,m_deltaSum,m_convDim);
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	gradBias<<<1,m_curNumFeature>>>(m_biasGrad,m_deltaSum,batch);
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);
}

void CConvLayerGPU::updateWeight(float mom,float alpha)
{
	int threadNum=min(1024,m_inputNumFeature*m_kernelDim*m_kernelDim);
	g_weightAndBiasAdd<<<m_curNumFeature,threadNum>>>(
												m_weight,
												m_weightGrad,
												m_vecWeight,
												m_bias,
												m_biasGrad,
												m_vecBias,
												m_kernelDim*m_kernelDim*m_inputNumFeature,
												mom,
												alpha);
	cudaError_t cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);
}

int CConvLayerGPU::initMem()
{
	m_weightLen=m_kernelDim*m_kernelDim*m_inputNumFeature*m_curNumFeature;
	DL_ASSER(m_weightLen!=0);

	cudaError_t cudaStat;

	unsigned const int weightSize=sizeof(double)*m_kernelDim*m_kernelDim*m_inputNumFeature*m_curNumFeature;

	cudaStat=cudaMalloc((void**)&m_weightTmp,weightSize*batch);
	CUDA_ERROR(cudaStat);

	unsigned const int convDataSize=sizeof(double)*m_convDim*m_convDim*batch*m_curNumFeature;

	cudaStat=cudaMalloc((void**)&m_delta,convDataSize);
	CUDA_ERROR(cudaStat);

	cudaStat=cudaMalloc((void**)&m_deltaSum,sizeof(double)*batch*m_curNumFeature);
	CUDA_ERROR(cudaStat);

	cudaStat=cudaMalloc((void**)&m_convData,convDataSize);
	CUDA_ERROR(cudaStat);

	cudaStat=cudaMalloc((void**)&m_convNoActiveData,convDataSize);
	CUDA_ERROR(cudaStat);

	return CLayerGPU::initMem();
}

void CConvLayerGPU::freeMem()
{
	GPU_FREE(m_delta);
	GPU_FREE(m_convData);
	GPU_FREE(m_convNoActiveData);
	GPU_FREE(m_weightTmp);
	GPU_FREE(m_deltaSum);

	CLayerGPU::freeMem();
}