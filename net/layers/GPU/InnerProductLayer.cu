#include"InnerProductLayer.cuh"

int CInnerProductLayerGPU::setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs)
{
	int inputSize=inputs[0]->size()/inputs[0]->num;
	m_weightLen=m_param.curNumFeature*inputSize;

	for(int i=0;i<inputs.size();i++)
	{
		outputs[i]->create(inputs[i]->num,m_param.curNumFeature,1,1);
	}
	return CLayerBaseGPU::setup(inputs,outputs);
}

//block<<<batch>>>
//thread<<<min(1024,numFeature)>>>
__global__ void feedForwardActive(precision* innerProductData,precision* bias,int numFeature)
{
	int index=blockIdx.x*numFeature;
	for(int id = 0; id < numFeature; id += blockDim.x) 
	{ 
		int idx = id + threadIdx.x; 
		if(idx < numFeature)
			innerProductData[index+idx]+=bias[idx];
	}
}

precision CInnerProductLayerGPU::feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	int inputSize=bottoms[0]->size()/bottoms[0]->num;

	cudaError_t cudaStat=cudaSuccess;
	int wLen=m_param.curNumFeature*inputSize;
	dim3 threads = min(1024, wLen); 
	dim3 blocks  = min(1024, (wLen + threads.x - 1) / threads.x); 
	
#ifndef FLOAT_TYPE
	CTools::cudaMatrixMulD(bottoms[0]->gpuData,bottoms[0]->num,inputSize,m_weight.gpuData,m_param.curNumFeature,inputSize,tops[0]->gpuData,m_param.curNumFeature,CTools::TRANSPOSE_Y);
#endif

	blocks=bottoms[0]->num;
	threads=min(1024,m_param.curNumFeature);
	feedForwardActive<<<blocks,threads>>>(tops[0]->gpuData,m_bias.gpuData,m_param.curNumFeature);
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	double finSum=getWeightCost(m_weight.gpuData,wLen);
	return finSum*m_param.lambda/2;
}

//blocks<<<batch,NumFeature>>>
//thread<<<inputNumFeature>>>
__global__ void dFullActive(
						  precision* deltaData,
						  precision* activeData
						  )
{
	int srcNo=blockIdx.x;
	int featureNo=threadIdx.x;
	int index=srcNo*blockDim.x+featureNo;
	deltaData[index]*=activeData[index];
}

int CInnerProductLayerGPU::backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms)
{
	int inputSize=bottoms[0]->size()/bottoms[0]->num;
	dFullActive<<<tops[0]->num,m_param.curNumFeature>>>(tops[0]->gpuDiff,
														tops[0]->gpuData);
	cudaError_t cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

#ifndef FLOAT_TYPE
	CTools::cudaMatrixMulD(tops[0]->gpuDiff,tops[0]->num,m_param.curNumFeature,m_weight.gpuData,m_param.curNumFeature,inputSize,bottoms[0]->gpuDiff,inputSize,CTools::NORMAL_XY);
#endif

	return getGrad(tops,bottoms);
}

int CInnerProductLayerGPU::getGrad(std::vector<Blob<precision>*>& tops,std::vector<Blob<precision>*>& bottoms)
{
	cudaError_t cudaStat;

	int inputSize=bottoms[0]->size()/bottoms[0]->num;
#ifndef FLOAT_TYPE
	CTools::cudaMatrixMulD(tops[0]->gpuDiff,bottoms[0]->num,m_param.curNumFeature,bottoms[0]->gpuData,bottoms[0]->num,inputSize,m_weightGrad,inputSize,CTools::TRANSPOSE_X);
#endif

	dim3 blocks(m_param.curNumFeature,inputSize);

	int wLen=m_param.curNumFeature*inputSize;
	dim3 threads2 = min(1024, wLen); 
	dim3 blocks2  = min(1024, (wLen + threads2.x - 1) / threads2.x); 

	fullWeightGrad<<<blocks2,threads2>>>(m_weightGrad,m_weight.gpuData,wLen,m_param.lambda,bottoms[0]->num);
	cudaStat=cudaDeviceSynchronize();
	CUDA_ERROR(cudaStat);

	this->computeBiasGrad(tops[0]->gpuDiff,m_biasGrad,bottoms[0]->num);
	return NET_SUCCESS;
}