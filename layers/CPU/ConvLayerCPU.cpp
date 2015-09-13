#include "ConvLayerCPU.h"

int CConvLayerCPU::initMem()
{
	m_weightLen=m_kernelDim*m_kernelDim*m_inputNumFeature*m_curNumFeature;
	DL_ASSER(m_weightLen!=0);

	unsigned const int convDataSize=m_convDim*m_convDim*batch*m_curNumFeature;

	m_delta=new precision[convDataSize];
	DL_ASSER(m_delta);

	m_convData=new precision[convDataSize];
	DL_ASSER(m_convData);

	m_convNoActiveData=new precision[convDataSize];
	DL_ASSER(m_convNoActiveData);

	return CLayerBaseCPU::initMem();
}


void CConvLayerCPU::freeMem()
{
	CPU_FREE(m_delta);
	CPU_FREE(m_convData);
	CPU_FREE(m_convNoActiveData);

	CLayerBaseCPU::freeMem();
}

struct convData
{
	precision* pSrc;
	precision* pNoActiveConv;
	precision* pActiveConv;
	precision* pWeight;
	precision* pBias;

	int kernelDim;
	int convDim;
	int inputNumFeature;
	int numFeature;
	int type;
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
		precision* pNoConvV=NULL;
		for(int i=row;i<end;)
		{
			pSrc=pData->pSrc+i*inputDataSize*inputNumFeature;
			pDest=pData->pActiveConv+i*conDim*conDim*numFeature;
			pNoConvV=pData->pNoActiveConv+i*conDim*conDim*numFeature;;
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
					precision* pNoConData=pNoConvV+j*conDim*conDim;
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
					res+=pData->pBias[j];
					pNoConData[i*conDim+g]=res;
					pConvData[i*conDim+g]=activeFunCPU(res,pData->type);
				}
			}
		}
	}
	return 0;
}

void    CConvLayerCPU::feedforward(double* srcData,DLparam& params)
{
	CProcessPool* pPP=CProcessPool::initInstance();
	convData cd1;
	cd1.inputNumFeature=m_inputNumFeature;
	cd1.numFeature=m_curNumFeature;
	cd1.kernelDim=m_kernelDim;
	cd1.convDim=m_convDim;
	cd1.pActiveConv=m_convData;
	cd1.pNoActiveConv=m_convNoActiveData;
	cd1.pWeight=m_weight;
	cd1.pBias=m_bias;
	cd1.pSrc=srcData;
	cd1.type=params.activeType;
	pPP->start(ConvThread,&cd1,batch);
	pPP->wait();

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
						kernel_i = kernelR - 1 - PK_MAX(0, edgeR - i);  
						src_i = PK_MAX(0, i - edgeR);  
						for (; kernel_i >= 0 && src_i < srcR; kernel_i--, src_i++)
						{  
							kernel_j = kernelC - 1 - PK_MAX(0, edgeC - j);  
							src_j =PK_MAX(0, j - edgeC);  
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

void CConvLayerCPU::backpropagation(double*preDelta,DLparam& params)
{
	CProcessPool* pPP=CProcessPool::initInstance();
	nolineData cd2;
	cd2.deltaS=m_delta;
	cd2.activeData=m_convNoActiveData;
	cd2.deltaE=m_delta;
	cd2.type=params.activeType;
	pPP->start(NolineProcessThread,&cd2,m_convDim*m_convDim*m_curNumFeature*batch);
	pPP->wait();

	if(preDelta!=NULL)
	{
		memset(preDelta,0,sizeof(double)*(m_convDim+m_kernelDim-1)*(m_convDim+m_kernelDim-1)*m_inputNumFeature*batch);

		fullConvData fcd;

		fcd.kernelDim=m_kernelDim;
		fcd.convDim=m_convDim;
		fcd.curNumFeature=m_curNumFeature;
		fcd.inputNumFeature=m_inputNumFeature;

		fcd.pDeltaD=preDelta;
		fcd.pDeltaS=m_delta;
		fcd.pWeight=m_weight;
		pPP->start(fullConvThread,&fcd,batch);
		pPP->wait();
	}
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

void CConvLayerCPU::getGrad(double* srcData)
{
	CProcessPool* pPP=CProcessPool::initInstance();
	conv2Data cd;
	
	cd.numberImages=batch;
	cd.lambda=m_lambda;
	cd.conDim=m_convDim;
	cd.kernelDim=m_kernelDim;
	cd.inputNumFeature=m_inputNumFeature;
	cd.curNumFeature=m_curNumFeature;

	cd.pSrc=srcData;
	cd.delta=m_delta;
	cd.pWeight=m_weight;
	cd.pWeightGrad=m_weightGrad;

	pPP->start(conv2ProcessThread,&cd,m_curNumFeature);
	pPP->wait();

	int convDim2=m_convDim*m_convDim;
	//ÕûºÏwb
	for(int i=0;i<m_curNumFeature;i++)
	{
		precision sum=0.0;
		for(int j=0;j<batch;j++)
			for(int g=0;g<convDim2;g++)
				sum+=m_delta[j*convDim2*m_curNumFeature+i*convDim2+g];
		m_biasGrad[i]=sum/batch;
	}
};


double  CConvLayerCPU::getCost(DLparam& params)
{
	return getWeightCost();
};