#include"ConvLayer.cuh"

int CConvLayerGPU::setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs)
{
	m_weightLen=m_param.convParam.kernelDim*m_param.convParam.kernelDim*inputs[0]->dataChannel*m_param.curNumFeature;
	DL_ASSER(m_weightLen!=0);

	cudaError_t cudaStat;

	unsigned const int weightSize=sizeof(precision)*m_weightLen;

	cudaStat=cudaMalloc((void**)&m_weightTmp,weightSize*inputs[0]->num);
	CUDA_ERROR(cudaStat);

	for(int i=0;i<inputs.size();i++)
	{
		int convDim=inputs[i]->dimWidth-m_param.convParam.kernelDim+1;
		outputs[i]->create(inputs[i]->num,m_param.curNumFeature,convDim,convDim);
	}

	cudaStat=cudaMalloc((void**)&m_deltaSum,sizeof(precision)*inputs[0]->num*m_param.curNumFeature);
	CUDA_ERROR(cudaStat);

	return CLayerBaseGPU::setup(inputs,outputs);
}

void CConvLayerGPU::freeMem()
{
	GPU_FREE(m_weightTmp);
	GPU_FREE(m_deltaSum);

	CLayerBaseGPU::freeMem();
}

//block<<<batch,m_curNumFeature>>>
//threads<<<min(1024,convDim*convDim)>>>
__global__ void feedForwardConvolution(
							precision* srcData,
							int srcDim,
							int inputNumFeature,
							precision* maskData,
							int maskDim,
							int numFeature,
							precision* destData,
							precision* pB
							)
{
	int srcNo=blockIdx.x;
	int featureNo=blockIdx.y;
	int conDim=srcDim-maskDim+1;
	int conDim2=conDim*conDim;
	int srcDim2=srcDim*srcDim;
	int maskDim2=maskDim*maskDim;
	
	precision* pSrc=srcData+srcNo*srcDim2*inputNumFeature;
	precision* pMask=maskData+featureNo*maskDim2*inputNumFeature;
	precision* pDest=destData+(srcNo*numFeature+featureNo)*conDim2;

	for(int idx=threadIdx.x;idx<conDim2;idx+=blockDim.x)
	{
		int convX=idx%conDim;
		int convY=idx/conDim;
		precision res=0.0;
		for(int k=0;k<inputNumFeature;k++)
		{
			precision* pX1=pSrc+k*srcDim2;
			precision* pX2=pMask+k*maskDim2;

			for(int x1=0;x1<maskDim;x1++)
			{
				for(int x2=0;x2<maskDim;x2++)			
					res+= pX1[(convY+x1)*srcDim+convX+x2] * pX2[x1*maskDim+x2];
			}
		}
		pDest[convY*conDim+convX]=res+pB[featureNo];
	}
}

precision CConvLayerGPU::feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	for(int i=0;i<bottoms.size();i++)
	{
		int convDim=bottoms[i]->dimWidth-m_param.convParam.kernelDim+1;
		dim3 blocks(bottoms[i]->num,m_param.curNumFeature);
		dim3 threads(min(1024,convDim*convDim));
		feedForwardConvolution<<<blocks,threads>>>
							(bottoms[i]->gpuData,
							 bottoms[i]->dimWidth,
							 bottoms[i]->dataChannel,
							 m_weight.gpuData,
							 m_param.convParam.kernelDim,
							 m_param.curNumFeature,
							 tops[i]->gpuData,
							 m_bias.gpuData);
	cudaError_t cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);
	}

	//计算权值损失函数
	int dataSize=m_param.curNumFeature*bottoms[0]->dataChannel*m_param.convParam.kernelDim*m_param.convParam.kernelDim;
	double finSum=getWeightCost(m_weight.gpuData,dataSize);
	return finSum*m_param.lambda/2;
}

//blocks<<<batch,NumFeature>>>
//thread<<<min(1024,convDim*convDim)>>>
__global__ void g_dConvActive(
						  precision* deltaData,
						  precision* convActiveData,
						  int convDim
						  )
{
	int convDim2=convDim*convDim;
	int index=blockIdx.x*convDim2*gridDim.y+blockIdx.y*convDim2;
	for(int idx=threadIdx.x;idx<convDim2;idx+=blockDim.x)
		deltaData[index+idx]*=convActiveData[index+idx];
}

//blocks<<<batch,inputNumFeature>>>
//threads<<<min(1024,curPoolDeltaDim*curPoolDeltaDim)>>>
__global__ void backDeltaFromConvToPool(
										precision* convDelta,
										int convDim,
										int convNumFeature,
										precision* convWeight,
										int maskDim,
										precision* poolDelta,
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
	
	precision* pConvDetal=convDelta+srcNo*convNumFeature*convDim2;
	precision* pPool=poolDelta+(srcNo*poolNumFeature+featureNo)*poolDim2;

	for(int idx=threadIdx.x;idx<dstSize2;idx+=blockDim.x)
	{
		int j=idx%dstSize;
		int i=idx/dstSize;  
		precision res=0.0;
		for(int g=0;g<convNumFeature;g++)
		{
			precision* pSrc=pConvDetal+g*convDim2;
			precision* pMask=convWeight+g*maskDim2*poolNumFeature+featureNo*maskDim2;  
			precision sum = 0.0; 

			kernel_i = maskDim - 1 - max(0, edgeSize - i);  
			src_i = max(0, i - edgeSize);  
			for (; kernel_i >= 0 && src_i < convDim; kernel_i--, src_i++)
			{  
				kernel_j = maskDim - 1 - max(0, edgeSize - j);  
				src_j =max(0, j - edgeSize);  
				for (; kernel_j >= 0 && src_j < convDim; kernel_j--, src_j++)
					sum += pSrc[src_i*convDim+src_j] * pMask[kernel_i*maskDim+kernel_j];  
			}            
			res+= sum; 		 
		} 
		pPool[i*dstSize+j]=res;
	}
}


int CConvLayerGPU::backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms)
{
	for(int i=0;i<tops.size();i++)
	{
		int convDim=tops[i]->dimWidth;
		dim3 blocks(tops[i]->num,m_param.curNumFeature);
		dim3 threads(min(1024,convDim*convDim));
		g_dConvActive<<<blocks,threads>>>(tops[i]->gpuDiff,
										  tops[i]->gpuData,
										  convDim
										  );
		cudaError_t cudaStat=cudaDeviceSynchronize();
		CUDA_ERROR(cudaStat);

		if(propagateDown[i])
		{
			dim3 blocks(tops[i]->num,bottoms[i]->dataChannel);
			threads=min(1024,bottoms[i]->dimWidth*bottoms[i]->dimWidth);

			backDeltaFromConvToPool<<<blocks,threads>>>(tops[i]->gpuDiff,
														convDim,
														m_param.curNumFeature,
														m_weight.gpuData,
												        m_param.convParam.kernelDim,
												        bottoms[i]->gpuDiff,
												        bottoms[i]->dataChannel);
			cudaStat=cudaDeviceSynchronize();
			 CUDA_ERROR(cudaStat);
		}
	}
	return this->getGrad(tops,bottoms);
}

//blocks<<<batch,m_curNumFeature>>>
//threads<<<WeightDim,WeightDim>>>
__global__ void gradWeight(precision* srcData,
							int srcDim,
							int inputNumFeature,
							precision* deltaData,
							int deltaDim,
							int numFeature,
							precision* weightGradData)
{
	int srcNo=blockIdx.x;
	int featureNo=blockIdx.y;
	int conDim=srcDim-deltaDim+1;
	int conDim2=conDim*conDim;
	int srcDim2=srcDim*srcDim;
	int deltaDim2=deltaDim*deltaDim;
	
	precision* pSrc=srcData+srcNo*srcDim2*inputNumFeature;
	precision* pMask=deltaData+srcNo*deltaDim2*gridDim.y;
	precision* pGrad=weightGradData+srcNo*numFeature*inputNumFeature*conDim2+featureNo*inputNumFeature*conDim2;

	int convX=threadIdx.x;
	int convY=threadIdx.y;
	
	for(int k=0;k<inputNumFeature;k++)
	{
		precision* pX1=pSrc+k*srcDim2;
		precision* pX2=pMask+featureNo*deltaDim2;
		precision res=0.0;

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
__global__ void gradWeight2(precision* srcData,
							int srcDim,
							int inputNumFeature,
							precision* deltaData,
							int deltaDim,
							int numFeature,
							precision* weightGradData)
{
	int srcNo=blockIdx.x;
	int featureNo=blockIdx.y;
	int conDim=srcDim-deltaDim+1;
	int conDim2=conDim*conDim;
	int srcDim2=srcDim*srcDim;
	int deltaDim2=deltaDim*deltaDim;
	
	precision* pSrc=srcData+srcNo*srcDim2*inputNumFeature;
	precision* pMask=deltaData+srcNo*deltaDim2*gridDim.y;
	precision* pGrad=weightGradData+srcNo*numFeature*inputNumFeature*conDim2+featureNo*inputNumFeature*conDim2;

	int convX=threadIdx.x;
	int convY=threadIdx.y;
	
	int k=blockIdx.z;
//	for(int k=0;k<inputNumFeature;k++)
//	{
		precision* pX1=pSrc+k*srcDim2;
		precision* pX2=pMask+featureNo*deltaDim2;
		precision res=0.0;

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
							precision* weightGrad,
							precision* weightTmp,
							precision* weight,
							int  weightDim,
							float lambda,
							int batch)
{
	int featureNo=blockIdx.x;
	int inputFeatureNo=blockIdx.y;
	int tidX=threadIdx.x;
	int tidY=threadIdx.y;

	int weightDim2=weightDim*weightDim;

	int index=(featureNo*gridDim.y+inputFeatureNo)*weightDim2+tidY*weightDim+tidX;
	precision tmp=0.0;
	
	int oneDataSize=gridDim.x*gridDim.y*weightDim2;
	for(int i=0;i<batch;i++)
		tmp+=weightTmp[index+i*oneDataSize];
	weightGrad[index]=tmp/batch+weight[index]*lambda;
}

//blocks<<<curNumFeature,batch>>>
//threads<<min(threadNum,deltaDim*deltaDim)>>>
//share<<threads*sizeof(precision>>>
__global__ void addDeltaSum(precision* delta,
							precision* deltaSum,
							int deltaDim)
{
	extern __shared__ precision _sum[];
	int featureNo=blockIdx.x;
	int imgNo=blockIdx.y;
	int tid=threadIdx.x;
	int deltaDim2=deltaDim*deltaDim;

	precision* pDelta=delta+imgNo*gridDim.x*deltaDim2+featureNo*deltaDim2;
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

int CConvLayerGPU::getGrad(std::vector<Blob<precision>*>& tops,std::vector<Blob<precision>*>& bottoms)
{
	cudaError_t cudaStat=cudaSuccess;
	for(int i=0;i<tops.size();i++)
	{
		dim3 blocks(tops[i]->num,m_param.curNumFeature,bottoms[i]->dataChannel);
		dim3 threads(m_param.convParam.kernelDim,m_param.convParam.kernelDim);

		gradWeight2<<<blocks,threads>>>(bottoms[i]->gpuData,
										bottoms[i]->dimWidth,
									    bottoms[i]->dataChannel,
										tops[i]->gpuDiff,
							            tops[i]->dimWidth,
										m_param.curNumFeature,
										m_weightTmp
										);
		cudaStat=cudaDeviceSynchronize();
		CUDA_ERROR(cudaStat);

		dim3 blocks2(m_param.curNumFeature,bottoms[i]->dataChannel);
		dim3 threads2(m_param.convParam.kernelDim,m_param.convParam.kernelDim);

		gradWeightAdd<<<blocks2,threads2>>>(m_weightGrad,
											m_weightTmp,
											m_weight.gpuData,
											m_param.convParam.kernelDim,
											m_param.lambda,
											tops[i]->num
											);
		cudaStat=cudaDeviceSynchronize();
		CUDA_ERROR(cudaStat);

		dim3 block3(m_param.curNumFeature,tops[i]->num);
		int threadNum=min(256,tops[i]->dimWidth*tops[i]->dimWidth);

		addDeltaSum<<<block3,threadNum,sizeof(precision)*threadNum>>>(tops[i]->gpuDiff,m_deltaSum,tops[i]->dimWidth);
		cudaStat=cudaDeviceSynchronize();
		CUDA_ERROR(cudaStat);

		this->computeBiasGrad(m_deltaSum,m_biasGrad,tops[i]->num);
	}
	return NET_SUCCESS;
}