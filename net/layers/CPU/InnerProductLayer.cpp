#include "InnerProductLayer.h"

int CInnerProductLayerCPU::setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs)
{
	int inputSize=inputs[0]->size()/inputs[0]->num;;
	m_weightLen=m_param.curNumFeature*inputSize;
	DL_ASSER(m_weightLen!=0);

	outputs[0]->create(inputs[0]->num,m_param.curNumFeature,1,1);

	return CLayerBaseCPU::setup(inputs,outputs);
}

struct fullData
{
	precision* pSrc;
	precision* pW;
	precision* pB;
	precision* pDest;

	int inputNumFeature;
	int curNumFeature;
	int batch;
};

unsigned int __stdcall fullConnectProcessThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int start=pUData->start;
	int end=pUData->end;
	fullData* pUserData=(fullData*)pUData->pUserData;
	precision* pW=pUserData->pW;
	precision* pSrc=pUserData->pSrc;
	precision* pDest=pUserData->pDest;
	precision* pB=pUserData->pB;
	int inputNumFeature=pUserData->inputNumFeature;
	int curNumFeature=pUserData->curNumFeature;
	int batch=pUserData->batch;
	for(int i=start;i<end;i++)
	{
		for(int j=0;j<batch;j++)
		{
			precision sum=0.0;
			for(int k=0;k<inputNumFeature;k++)
				sum+=pW[i*inputNumFeature+k]*pSrc[j*inputNumFeature+k];
			pDest[j*curNumFeature+i]=sum+pB[i];
		}

	}
	return 0;
}

precision CInnerProductLayerCPU::feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	int inputSize=bottoms[0]->size()/bottoms[0]->num;
	int num=bottoms[0]->num;

	CProcessPool* pPP=CProcessPool::initInstance();

	fullData fd;
	fd.pB=m_bias.cpuData;
	fd.pW=m_weight.cpuData;

	fd.pSrc=bottoms[0]->cpuData;
	fd.inputNumFeature=inputSize;
	fd.curNumFeature=m_param.curNumFeature;
	fd.batch=num;
	fd.pDest=tops[0]->cpuData;

	pPP->start(fullConnectProcessThread,&fd,m_param.curNumFeature);
	pPP->wait();

	return getWeightCost();

};

int CInnerProductLayerCPU::backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms)
{
	int num=bottoms[0]->num;
	int inputSize=bottoms[0]->size()/bottoms[0]->num;

	CProcessPool* pPP=CProcessPool::initInstance();
	nolineData cd2;
	cd2.deltaS=tops[0]->cpuDiff;
	cd2.activeData=tops[0]->cpuData;
	cd2.deltaE=tops[0]->cpuDiff;
	pPP->start(NolineProcessThread,&cd2,m_param.curNumFeature*num);
	pPP->wait();

	matrixMul(tops[0]->cpuDiff,num,m_param.curNumFeature,m_weight.cpuData,m_param.curNumFeature,inputSize,bottoms[0]->cpuDiff);
	return getGrad(tops,bottoms);
};

struct fullData2
{
	precision* pSrc;
	precision* pDelta;
	precision* pWeight;
	precision* pWeightGrad;

	int inputNumFeature;
	int curNumFeature;
	int numImages;
	float lambda;
};

unsigned int __stdcall d_fullProcessThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int start=pUData->start;
	int end=pUData->end;
	fullData2* pUserData=(fullData2*)pUData->pUserData;
	precision* pWeight=pUserData->pWeight;
	precision* pSrc=pUserData->pSrc;
	precision* pWeightGrad=pUserData->pWeightGrad;
	precision* pDelta=pUserData->pDelta;

	for(int i=start;i<end;i++)
	{
		for(int j=0;j<pUserData->inputNumFeature;j++)
		{
			double sum=0.0;
			for(int k=0;k<pUserData->numImages;k++)
				sum+=pDelta[k*pUserData->curNumFeature+i]*pSrc[k*pUserData->inputNumFeature+j];
			pWeightGrad[i*pUserData->inputNumFeature+j]=sum/pUserData->numImages+pWeight[i*pUserData->inputNumFeature+j]*pUserData->lambda;
		}
	}
	return 0;
}

int CInnerProductLayerCPU::getGrad(std::vector<Blob<precision>*>& tops,std::vector<Blob<precision>*>& bottoms)
{
	int num=bottoms[0]->num;
	int inputSize=bottoms[0]->size()/bottoms[0]->num;

	CProcessPool* pPP=CProcessPool::initInstance();
	fullData2 fd2;
	fd2.lambda=m_param.lambda;
	fd2.numImages=num;
	fd2.inputNumFeature=inputSize;
	fd2.curNumFeature=m_param.curNumFeature;

	fd2.pSrc=bottoms[0]->cpuData;
	fd2.pWeightGrad=m_weightGrad;
	fd2.pWeight=m_weight.cpuData;
	fd2.pDelta=tops[0]->cpuDiff;

	pPP->start(d_fullProcessThread,&fd2,m_param.curNumFeature);
	pPP->wait();

	for(int i=0;i<m_param.curNumFeature;i++)
	{
		precision sum=0;
		for(int j=0;j<num;j++)
			sum+=tops[0]->cpuDiff[j*m_param.curNumFeature+i];
		m_biasGrad[i]=sum/num;
	}
	return 0;
};