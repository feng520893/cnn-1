#include "ConvLayer.h"

int CConvLayerCPU::setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs)
{
	m_weightLen=m_param.convParam.kernelDim*m_param.convParam.kernelDim*inputs[0]->dataChannel*m_param.curNumFeature;
	DL_ASSER(m_weightLen!=0);

	for(int i=0;i<inputs.size();i++)
	{
		int convDim=inputs[i]->dimWidth-m_param.convParam.kernelDim+1;
		outputs[i]->create(inputs[i]->num,m_param.curNumFeature,convDim,convDim);
	}

	return CLayerBaseCPU::setup(inputs,outputs);
}

struct convData
{
	precision* pSrc;
	precision* pActiveConv;
	precision* pWeight;
	precision* pBias;

	int kernelDim;
	int convDim;
	int inputNumFeature;
	int numFeature;
};

unsigned int __stdcall ConvThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int row=pUData->start;
	int end=pUData->end;
	convData* pData=(convData*)pUData->pUserData;
	int offset=0;
	int kernel2=pData->kernelDim*pData->kernelDim;
	int maskDim=pData->kernelDim;
	int conDim=pData->convDim;
	int inputDim=pData->convDim+maskDim-1;
	int inputDataSize=inputDim*inputDim;
	int inputNumFeature=pData->inputNumFeature;
	int numFeature=pData->numFeature;

	while(1)
	{
		precision* pSrc=NULL;
		precision* pDest=NULL;
		for(int i=row;i<end;)
		{
			pSrc=pData->pSrc+i*inputDataSize*inputNumFeature;
			pDest=pData->pActiveConv+i*conDim*conDim*numFeature;
			++row;
			break;
		}
		if(pSrc==NULL)
			break;

		for(int j=0;j<numFeature;j++)
		{
			for(int i=0;i<conDim;i++)
			{
				for(int g=0;g<conDim;g++)
				{
					precision res=0.0;
					precision* pConvData=pDest+j*conDim*conDim;
					for(int k=0;k<inputNumFeature;k++)
					{
						offset=j*kernel2*inputNumFeature+k*kernel2;
						double* pX1=pSrc+k*inputDataSize;
						double* pX2=pData->pWeight+offset;

						for(int x1=0;x1<maskDim;x1++)
						{
							for(int x2=0;x2<maskDim;x2++)			
							{
								res+= pX1[(i+x1)*inputDim+g+x2] * pX2[x1*maskDim+x2];
							}
						}

					}
					pConvData[i*conDim+g]=res+pData->pBias[j];
				}
			}
		}
	}
	return 0;
}

precision CConvLayerCPU::feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	CProcessPool* pPP=CProcessPool::initInstance();
	convData cd;
	for(int i=0;i<bottoms.size();i++)
	{
		cd.inputNumFeature=bottoms[i]->dataChannel;
		cd.numFeature=m_param.curNumFeature;
		cd.kernelDim=m_param.convParam.kernelDim;
		cd.convDim=tops[i]->dimWidth;
		cd.pActiveConv=tops[i]->cpuData;
		cd.pWeight=m_weight.cpuData;
		cd.pBias=m_bias.cpuData;
		cd.pSrc=bottoms[i]->cpuData;
		pPP->start(ConvThread,&cd,bottoms[i]->num);
		pPP->wait();
	}

	return getWeightCost();
};


struct fullConvData
{
	precision* pDeltaS;
	precision* pDeltaD;
	precision* pWeight;

	int kernelDim;
	int convDim;
	int curNumFeature;
	int inputNumFeature;

};

unsigned int __stdcall fullConvThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int row=pUData->start;
	int end=pUData->end;
	fullConvData* pUserData=(fullConvData*)pUData->pUserData;

	int outputDim=pUserData->convDim+pUserData->kernelDim-1;
	int outputDim2=pUserData->convDim;
	int kernelDim2=pUserData->kernelDim*pUserData->kernelDim;
	while(1)
	{
		precision* pDeltaS=NULL;
		precision* pDeltaD=NULL;
		int* pMaxIndex=NULL;
		for(int i=row;i<end;)
		{
			pDeltaS=pUserData->pDeltaS+i*outputDim2*outputDim2*pUserData->curNumFeature;
			pDeltaD=pUserData->pDeltaD+i*outputDim*outputDim*pUserData->inputNumFeature;
			row++;
			break;
		}
		if(pDeltaS==NULL)
		{
			break;
		}


		int srcR = outputDim2;//pDeltaS->at(0).Row;
		int kernelR =pUserData->kernelDim;
		int srcC = outputDim2;//pDeltaS->at(0).Col;
		int kernelC =pUserData->kernelDim;
		int dstR = srcR + kernelR - 1;  
		int dstC = srcC + kernelC - 1;        
		int edgeR = kernelR - 1;  
		int edgeC = kernelC - 1;  
		int kernel_i=0,src_i=0,kernel_j=0,src_j=0;

		for(int i=0;i<pUserData->inputNumFeature;i++)
		{
			precision* pDest=pDeltaD+i*outputDim*outputDim;
			for(int j=0;j<pUserData->curNumFeature;j++)
			{
				precision* pSrc=pDeltaS+j*outputDim2*outputDim2;
				precision* pMask=pUserData->pWeight+j*kernelDim2*pUserData->inputNumFeature+i*kernelDim2;
				for (int i = 0; i < dstR; i++) 
				{    
					for (int j = 0; j < dstC; j++) 
					{       
						double sum = 0;  
						kernel_i = kernelR - 1 - max(0, edgeR - i);  
						src_i = max(0, i - edgeR);  
						for (; kernel_i >= 0 && src_i < srcR; kernel_i--, src_i++)
						{  
							kernel_j = kernelC - 1 - max(0, edgeC - j);  
							src_j =max(0, j - edgeC);  
							for (; kernel_j >= 0 && src_j < srcC; kernel_j--, src_j++)
								sum += pSrc[src_i*srcC+src_j] * pMask[kernel_i*kernelC+kernel_j];  
						}             
						pDest[i*dstC+j] += sum;  
					}  
				} 
			}
		}
	}
	return 0;
}

int CConvLayerCPU::backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms)
{
	CProcessPool* pPP=CProcessPool::initInstance();
	for(int i=0;i<tops.size();i++)
	{
		nolineData nd;
		nd.deltaS=tops[i]->cpuDiff;
		nd.activeData=tops[i]->cpuData;
		nd.deltaE=tops[i]->cpuDiff;
		int convDim=tops[i]->dimWidth;
		pPP->start(NolineProcessThread,&nd,convDim*convDim*m_param.curNumFeature*tops[i]->num);
		pPP->wait();
		
		if(propagateDown[i])
		{
			memset(bottoms[i]->cpuDiff,0,sizeof(precision)*bottoms[i]->size());
			
			fullConvData fcd;
			fcd.kernelDim=m_param.convParam.kernelDim;
			fcd.convDim=convDim;
			fcd.curNumFeature=m_param.curNumFeature;
			fcd.inputNumFeature=bottoms[i]->dataChannel;
			fcd.pDeltaD=bottoms[i]->cpuDiff;
			fcd.pDeltaS=tops[i]->cpuDiff;
			fcd.pWeight=m_weight.cpuData;
			pPP->start(fullConvThread,&fcd,tops[i]->num);
			pPP->wait();
		}
	}
	return getGrad(tops,bottoms);
};


struct conv2Data
{
	precision* delta;
	precision* pSrc;
	precision* pWeightGrad;
	precision* pWeight;

	int kernelDim;
	int numberImages;
	int conDim;
	int inputNumFeature;
	int curNumFeature;
	float lambda;
};

unsigned int __stdcall conv2ProcessThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int row=pUData->start;
	int end=pUData->end;
	conv2Data* pUserData=(conv2Data*)pUData->pUserData;

	int conDim=pUserData->kernelDim;
	int convDim2=conDim*conDim;
	int srcDim=pUserData->conDim+pUserData->kernelDim-1;
	int srcDim2=srcDim*srcDim;
	int kernelDim=pUserData->conDim;
	int kernelDim2=kernelDim*kernelDim;
	int numImages=pUserData->numberImages;

	while(1)
	{
		precision* wc_grad=NULL;
		for(int i=row;i<end;)
		{
			wc_grad=pUserData->pWeightGrad+i*convDim2*pUserData->inputNumFeature;
			break;
		}
		if(wc_grad==NULL)
		{
			break;
		}
		for(int k=0;k<pUserData->inputNumFeature;k++)
		{
			precision* pData=wc_grad+k*convDim2;

			for(int i=0;i<conDim;i++)
			{
				for(int j=0;j<conDim;j++)
				{
					precision res=0.0;
					for(int h=0;h<numImages;h++)  
					{
						precision* pX1=pUserData->pSrc+h*srcDim2*pUserData->inputNumFeature+k*srcDim2;
						precision* pX2=pUserData->delta+h*kernelDim2*pUserData->curNumFeature+row*kernelDim2;
						
						for(int x1=0;x1<kernelDim;x1++)
						{
							for(int x2=0;x2<kernelDim;x2++)			
							{
								res+= pX1[(i+x1)*srcDim+j+x2] * pX2[x1*kernelDim+x2];
							}
						}
					}
					pData[i*conDim+j]=res;
				}
			}
		}

		precision* pW=pUserData->pWeight+row*convDim2*pUserData->inputNumFeature;
		int leng=convDim2*pUserData->inputNumFeature;
		for(int j=0;j<leng;j++)
			wc_grad[j]=wc_grad[j]/numImages+pW[j]*pUserData->lambda;
		++row;
	}
	return 0;
}

int CConvLayerCPU::getGrad(std::vector<Blob<precision>*>& tops,std::vector<Blob<precision>*>& bottoms)
{
	CProcessPool* pPP=CProcessPool::initInstance();
	conv2Data cd;
	int convDim=tops[0]->dimWidth;
	cd.numberImages=tops[0]->num;
	cd.lambda=m_param.lambda;
	cd.conDim=convDim;
	cd.kernelDim=m_param.convParam.kernelDim;
	cd.inputNumFeature=bottoms[0]->dataChannel;
	cd.curNumFeature=m_param.curNumFeature;

	cd.pSrc=bottoms[0]->cpuData;
	cd.delta=tops[0]->cpuDiff;
	cd.pWeight=m_weight.cpuData;
	cd.pWeightGrad=m_weightGrad;

	pPP->start(conv2ProcessThread,&cd,m_param.curNumFeature);
	pPP->wait();

	int convDim2=convDim*convDim;
	//ÕûºÏwb
	for(int i=0;i<m_param.curNumFeature;i++)
	{
		precision sum=0.0;
		for(int j=0;j<tops[0]->num;j++)
			for(int g=0;g<convDim2;g++)
				sum+=tops[0]->cpuDiff[j*convDim2*m_param.curNumFeature+i*convDim2+g];
		m_biasGrad[i]=sum/tops[0]->num;
	}
	return 0;
};