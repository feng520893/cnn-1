#include "FullLayerCPU.h"



int CFullLayerCPU::initMem()
{
	m_weightLen=m_curNumFeature*m_inputNumFeature;

	DL_ASSER(m_weightLen!=0);

	m_delta=new precision[m_curNumFeature*batch];
	DL_ASSER(m_delta);

	m_afterDropWeight=new precision[m_inputNumFeature*m_curNumFeature];
	DL_ASSER(m_afterDropWeight);

	m_dropProbability=new float[m_inputNumFeature*m_curNumFeature];
	DL_ASSER(m_dropProbability);

	m_fullData=new precision[m_curNumFeature*batch];
	DL_ASSER(m_fullData);

	m_fullNoActiveData=new precision[m_curNumFeature*batch];
	DL_ASSER(m_fullNoActiveData);

	return CLayerBaseCPU::initMem();
}


void CFullLayerCPU::freeMem()
{
	CPU_FREE(m_fullData);
	CPU_FREE(m_fullNoActiveData);
	CPU_FREE(m_afterDropWeight);
	CPU_FREE(m_dropProbability);
	CLayerBaseCPU::freeMem();
}

struct fullData
{
	precision* pSrc;
	precision* pW;
	precision* pB;
	precision* pDest;
	precision* pNoDest;
	int inputNumFeature;
	int curNumFeature;
	int batch;
	int type;
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
	precision* pNoDest=pUserData->pNoDest;
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
			pNoDest[j*curNumFeature+i]=sum;
			pDest[j*curNumFeature+i]=activeFunCPU(sum+pB[i],pUserData->type);
		}

	}
	return 0;
}

struct dropWeightData
{
	precision* weights;
	precision* afterWeights;
	float* m_dropProbability;
	float dropRate;
	bool pred;
};

unsigned int __stdcall GetAfterDropWeightsThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int end=pUData->end;
	dropWeightData* pUserData=(dropWeightData*)pUData->pUserData;
	for(int tid=pUData->start;tid<end;tid++)
	{
		if(!pUserData->pred)
		{
			float rate=(float)rand() / RAND_MAX;
			if(pUserData->dropRate == 0||rate>pUserData->dropRate)
			{
				pUserData->m_dropProbability[tid]=1;
				pUserData->afterWeights[tid]=pUserData->weights[tid];
			}
			else
				pUserData->afterWeights[tid]=pUserData->m_dropProbability[tid]=0;
		}
		else
			pUserData->afterWeights[tid]=pUserData->weights[tid]*(1-pUserData->dropRate);
	}
	return 0;
}

void CFullLayerCPU::feedforward(double* srcData,DLparam& params)
{
/*
	pk::rand(m_dropProbability,m_curNumFeature*m_inputNumFeature);

	for(int i=0;i<m_curNumFeature*m_inputNumFeature;i++)
	{
		if(m_dropRate==0||m_dropProbability[i]>m_dropRate)
			m_dropProbability[i]=1;
		else
			m_dropProbability[i]=0;
		if(params.pred)
			m_afterDropWeight[i]=m_weight[i]*(1-m_dropRate);
		else
			m_afterDropWeight[i]=m_weight[i]*m_dropProbability[i];
	}*/

	srand(time(NULL));

	CProcessPool* pPP=CProcessPool::initInstance();

	dropWeightData dwd;
	dwd.afterWeights=m_afterDropWeight;
	dwd.weights=m_weight;
	dwd.m_dropProbability=m_dropProbability;
	dwd.dropRate=m_dropRate;
	dwd.pred=params.pred;

	pPP->start(GetAfterDropWeightsThread,&dwd,m_weightLen);
	pPP->wait();


	fullData fd;
	fd.type=params.activeType;
	fd.pB=m_bias;
	fd.pW=m_afterDropWeight;

	fd.pSrc=srcData;
	fd.inputNumFeature=m_inputNumFeature;
	fd.curNumFeature=m_curNumFeature;
	
	fd.batch=batch;
	fd.pDest=m_fullData;
	fd.pNoDest=m_fullNoActiveData;

	pPP->start(fullConnectProcessThread,&fd,m_curNumFeature);
	pPP->wait();

};

void CFullLayerCPU::backpropagation(precision*preDelta,DLparam& params)
{
	CProcessPool* pPP=CProcessPool::initInstance();
	nolineData cd2;
	cd2.deltaS=m_delta;
	cd2.activeData=m_fullNoActiveData;
	cd2.deltaE=m_delta;
	cd2.type=params.activeType;
	pPP->start(NolineProcessThread,&cd2,m_curNumFeature*batch);
	pPP->wait();

	matrixMul(m_delta,batch,m_curNumFeature,m_afterDropWeight,m_curNumFeature,m_inputNumFeature,preDelta);
};

struct fullData2
{
	precision* pSrc;
	precision* pDelta;
	float* pDropProbability;
	precision* pWeight;
	precision* pWeightGrad;

	int inputNumFeature;
	int curNumFeature;
	int numImages;
	float lambda;
	float dropRate;
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

	float* pDropProbability=pUserData->pDropProbability;

	for(int i=start;i<end;i++)
	{
		for(int j=0;j<pUserData->inputNumFeature;j++)
		{
			double sum=0.0;
			for(int k=0;k<pUserData->numImages;k++)
				sum+=pDelta[k*pUserData->curNumFeature+i]*pSrc[k*pUserData->inputNumFeature+j];
			pWeightGrad[i*pUserData->inputNumFeature+j]=sum/pUserData->numImages+pWeight[i*pUserData->inputNumFeature+j]*pUserData->lambda;
			if(pUserData->dropRate>0.0)
				pWeightGrad[i*pUserData->inputNumFeature+j]*=pDropProbability[i*pUserData->inputNumFeature+j];
		}
	}
	return 0;
}

void CFullLayerCPU::getGrad(double* srcData)
{

	CProcessPool* pPP=CProcessPool::initInstance();
	fullData2 fd2;
	fd2.lambda=m_lambda;
	fd2.numImages=batch;
	fd2.dropRate=m_dropRate;
	fd2.inputNumFeature=m_inputNumFeature;
	fd2.curNumFeature=m_curNumFeature;

	fd2.pSrc=srcData;
	fd2.pDropProbability=m_dropProbability;
	fd2.pWeightGrad=m_weightGrad;
	fd2.pWeight=m_weight;
	fd2.pDelta=m_delta;

	pPP->start(d_fullProcessThread,&fd2,m_curNumFeature);
	pPP->wait();

	for(int i=0;i<m_curNumFeature;i++)
	{
		precision sum=0;
		for(int j=0;j<batch;j++)
			sum+=m_delta[j*m_curNumFeature+i];
		m_biasGrad[i]=sum/batch;
	}

};

double CFullLayerCPU::getCost(DLparam& params)
{
	return getWeightCost();
};