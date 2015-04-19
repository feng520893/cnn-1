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


void CConvLayer::feedforward(double* srcData,int activeType)
{
	dim3 blocks(batch,m_curNumFeature);
	dim3 threads(min(1024,m_convDim*m_convDim));
	feedForwardConvolution<<<blocks,threads>>>
							(srcData,
							 m_convDim+m_maskDim-1,
							 m_inputNumFeature,
							 m_weight,
							 m_maskDim,
							 m_curNumFeature,
							 m_convData,
							 m_convNoActiveData,
							 m_bias,
							 activeType);

	cudaDeviceSynchronize();
	 
	int poolDim=m_convDim/m_poolArea;
	threads=min(1024,poolDim*poolDim);
	maxPool<<<blocks,threads>>>
							(m_convData,
							 m_convDim,
							 m_curNumFeature,
							 m_poolData,
							 m_maxIndexData,
							 m_poolArea);
	cudaDeviceSynchronize();
}

double CConvLayer::getCost()
{
	int dataSize=m_curNumFeature*m_inputNumFeature*m_maskDim*m_maskDim;
	double finSum=getWeightCost(m_weight,dataSize);
	return finSum*m_lambda/2;
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


int CConvLayer::backpropagation(double*preDelta,int activeType)
{
	cudaError_t cudaStat;

	cudaStat=cudaMemset(m_delta,0,sizeof(double)*m_convDim*m_convDim*batch*m_curNumFeature);

	if(cudaStat != cudaSuccess) 
	{ 
		printf ("device memory cudaMemset failed\n"); 
		exit(0); 
	}

	dim3 blocks(batch,m_curNumFeature);
	int poolDim=m_convDim/m_poolArea;
	dim3 threads(min(1024,poolDim*poolDim));

	backDeltaFromMaxPoolToConv<<<blocks,threads>>>(m_delta,
												   m_convDim,
												   m_curNumFeature,
												   m_poolDelta,
												   m_maxIndexData,
												   m_poolArea
												   );
	cudaDeviceSynchronize();

	threads=min(1024,m_convDim*m_convDim);
	g_dConvActive<<<blocks,threads>>>(m_delta,
									  m_convNoActiveData,
									  m_convDim,
									  activeType
									  );
	cudaDeviceSynchronize();

	if(preDelta!=NULL)
	{
		dim3 blocks(batch,m_inputNumFeature);
		int prePoolDim=m_convDim+m_maskDim-1;
		threads=min(1024,prePoolDim*prePoolDim);

		backDeltaFromConvToPool<<<blocks,threads>>>(m_delta,
													m_convDim,
													m_curNumFeature,
													m_weight,
											        m_maskDim,
											        preDelta,
											        m_inputNumFeature);
		 cudaDeviceSynchronize();
	}
	return 0;
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


//block3<<<curNumFeature>>>
//threads<<deltaDim*deltaDim>>>
//share<<deltaDim*deltaDim*sizeof(double>>>
__global__ void gradBias(
						 double* biasGrad,
						 double* delta,
						 int deltaDim,
						 int curNumFeature,
						 int batch
						 )
{
	extern __shared__ double _sum[];
	_sum[threadIdx.x] = 0.0; 
 	int deltaDim2 = deltaDim * deltaDim;  
 	for(int i = 0; i < batch;i++) 
	{ 
		int index=i*curNumFeature*deltaDim2+blockIdx.x*deltaDim2+threadIdx.x;
		_sum[threadIdx.x] += delta[index]; 
 	} 
	__syncthreads(); 

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
		biasGrad[blockIdx.x]=_sum[0]/batch;
}


void CConvLayer::getGrad(double* srcData)
{
	dim3 blocks(batch,m_curNumFeature,m_inputNumFeature);
	dim3 threads(m_maskDim,m_maskDim);
	int srcDim=m_convDim+m_maskDim-1;

	gradWeight2<<<blocks,threads>>>(srcData,
								   srcDim,
							       m_inputNumFeature,
							       m_delta,
							       m_convDim,
							       m_curNumFeature,
							       m_weightTmp
								   );
	cudaDeviceSynchronize();

	dim3 blocks2(m_curNumFeature,m_inputNumFeature);
	dim3 threads2(m_maskDim,m_maskDim);

	gradWeightAdd<<<blocks2,threads2>>>(
								m_weightGrad,
							    m_weightTmp,
							    m_weight,
								m_maskDim,
							    m_lambda,
							    batch);

	cudaDeviceSynchronize();

	gradBias<<<m_curNumFeature,m_convDim*m_convDim,sizeof(double)*m_convDim*m_convDim>>>(m_biasGrad,
																			m_delta,
																			m_convDim,
																			m_curNumFeature,
																			batch
																			);
	cudaDeviceSynchronize();
}

void CConvLayer::updateWeight(float mom,float alpha)
{
	int threadNum=min(1024,m_inputNumFeature*m_maskDim*m_maskDim);
	g_weightAndBiasAdd<<<m_curNumFeature,threadNum>>>(
												m_weight,
												m_weightGrad,
												m_vecWeight,
												m_bias,
												m_biasGrad,
												m_vecBias,
												m_maskDim*m_maskDim*m_inputNumFeature,
												mom,
												alpha);
	cudaDeviceSynchronize();

}

int CConvLayer::initMem()
{
	m_weightLen=m_maskDim*m_maskDim*m_inputNumFeature*m_curNumFeature;

	cudaError_t cudaStat;

	unsigned const int weightSize=sizeof(double)*m_maskDim*m_maskDim*m_inputNumFeature*m_curNumFeature;

	cudaStat=cudaMalloc((void**)&m_weightTmp,weightSize*batch);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMalloc failed\n");  
		freeMem();
		return -1;
	}

	unsigned const int convDataSize=sizeof(double)*m_convDim*m_convDim*batch*m_curNumFeature;

	cudaStat=cudaMalloc((void**)&m_delta,convDataSize);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMalloc failed\n");  
		freeMem();
		return -1;
	}

	cudaStat=cudaMalloc((void**)&m_convData,convDataSize);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMalloc failed\n");  
		freeMem();
		return -1;
	}

	cudaStat=cudaMalloc((void**)&m_convNoActiveData,convDataSize);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMalloc failed\n");  
		freeMem();
		return -1;
	}

	cudaStat=cudaMalloc((void**)&m_poolDelta,convDataSize/m_poolArea/m_poolArea);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMalloc failed\n");  
		freeMem();
		return -1;
	}

	cudaStat=cudaMalloc((void**)&m_poolData,convDataSize/m_poolArea/m_poolArea);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMalloc failed\n");  
		freeMem();
		return -1;
	}

	cudaStat=cudaMalloc((void**)&m_maxIndexData,sizeof(int)*m_convDim*m_convDim*batch*m_curNumFeature/m_poolArea/m_poolArea);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMalloc failed\n");  
		freeMem();
		return -1;
	}
	return CLayer::initMem();
}

void CConvLayer::freeMem()
{
	GPU_FREE(m_delta);
	GPU_FREE(m_poolData);
	GPU_FREE(m_convData);
	GPU_FREE(m_convNoActiveData);
	GPU_FREE(m_poolDelta);
	GPU_FREE(m_weightTmp);
	GPU_FREE(m_maxIndexData);

	CLayer::freeMem();
}