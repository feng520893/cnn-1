#include "SoftmaxLayerCPU.h"



int CSoftmaxLayerCPU::initMem()
{
	m_weightLen=m_curNumFeature*m_inputNumFeature;
	DL_ASSER(m_weightLen!=0);

	m_predDatas=new precision[m_curNumFeature*batch];
	DL_ASSER(m_predDatas);

	m_trueDatas=new precision[m_curNumFeature*batch];
	DL_ASSER(m_trueDatas);

	m_delta=new precision[m_curNumFeature*batch];
	DL_ASSER(m_delta);

	return CLayerBaseCPU::initMem();
}


void CSoftmaxLayerCPU::freeMem()
{
	CPU_FREE(m_predDatas);
	CPU_FREE(m_trueDatas);
	CLayerBaseCPU::freeMem();
}

struct softmaxData
{
	double* pSrc;
	double* pW;
	double* pB;
	double* pDest;
	int inputNumFeature;
	int numFeature;
	int batch;
};

unsigned int __stdcall softmaxConnectProcessThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int start=pUData->start;
	int end=pUData->end;
	softmaxData* pUserData=(softmaxData*)pUData->pUserData;
	double* pW=pUserData->pW;
	double* pSrc=pUserData->pSrc;
	double* pDest=pUserData->pDest;
	double* pB=pUserData->pB;
	int inputNumFeature=pUserData->inputNumFeature;
	int batch=pUserData->batch;
	int curNumFeature=pUserData->numFeature;
	for(int i=start;i<end;i++)
	{
		for(int j=0;j<batch;j++)
		{
			double sum=0.0;
			for(int k=0;k<inputNumFeature;k++)
				sum+=pW[i*inputNumFeature+k]*pSrc[j*inputNumFeature+k];
			pDest[j*curNumFeature+i]=sum+pB[i];
		}

	}
	return 0;
}

void CSoftmaxLayerCPU::feedforward(double* srcData,DLparam& params)
{
	softmaxData fd;
	fd.pB=m_bias;
	fd.pW=m_weight;

	fd.pSrc=srcData;
	fd.inputNumFeature=m_inputNumFeature;
	fd.numFeature=m_curNumFeature;
	
	fd.batch=batch;
	fd.pDest=m_predDatas;

	CProcessPool* pPP=CProcessPool::initInstance();

	pPP->start(softmaxConnectProcessThread,&fd,m_curNumFeature);
	pPP->wait();

	double * tmpMax=new double[batch];

	for(int i=0;i<batch;i++)
	{
		double max=0.0;
		for(int j=0;j<m_curNumFeature;j++)
		{
			if(m_predDatas[i*m_curNumFeature+j]>max)
				max=m_predDatas[i*m_curNumFeature+j];
		}
		tmpMax[i]=max;
	}

	//º∆À„∏≈¬ 
	for(int i=0;i<batch;i++)
	{
		double tSum=0.0;
		for(int j=0;j<m_curNumFeature;j++)
		{
			m_predDatas[i*m_curNumFeature+j]-=tmpMax[i];
			m_predDatas[i*m_curNumFeature+j]=exp(m_predDatas[i*m_curNumFeature+j]);
			tSum+=m_predDatas[i*m_curNumFeature+j];
		}
		for(int j=0;j<m_curNumFeature;j++)
			m_predDatas[i*m_curNumFeature+j]/=tSum;
	}

	delete [] tmpMax;

	if(params.pred)
	{
		for(int i=0;i<batch;i++)
		{
			double max=0.0;
			int index=0;
			for(int j=0;j<m_curNumFeature;j++)
			{
				if(m_predDatas[i*m_curNumFeature+j]>max)
				{
					index=j;
					max=m_predDatas[i*m_curNumFeature+j];
				}
			}
			params.predData[i]=index;
		}
	}
};

void CSoftmaxLayerCPU::backpropagation(double*preDelta,DLparam& params)
{
	memset(m_trueDatas,0,sizeof(precision)*m_curNumFeature*batch);
	for(int i=0;i<params.labels.size();i++)
		m_trueDatas[i*m_curNumFeature+params.labels[i]]=1;

	for(int i=0;i<m_curNumFeature*batch;i++)
		m_delta[i]=-(m_trueDatas[i]-m_predDatas[i]);
	matrixMul(m_delta,batch,m_curNumFeature,m_weight,m_curNumFeature,m_inputNumFeature,preDelta);
};


struct softmaxData2
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

unsigned int __stdcall d_softmaxProcessThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int start=pUData->start;
	int end=pUData->end;
	softmaxData2* pUserData=(softmaxData2*)pUData->pUserData;
	precision* pW=pUserData->pWeight;
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
			pWeightGrad[i*pUserData->inputNumFeature+j]=sum/pUserData->numImages+pW[i*pUserData->inputNumFeature+j]*pUserData->lambda;
		}
	}
	return 0;
}

void CSoftmaxLayerCPU::getGrad(double* srcData)
{
	CProcessPool* pPP=CProcessPool::initInstance();
	softmaxData2 sd;
	sd.lambda=m_lambda;
	sd.numImages=batch;
	sd.inputNumFeature=m_inputNumFeature;
	sd.curNumFeature=m_curNumFeature;

	sd.pSrc=srcData;
	sd.pWeightGrad=m_weightGrad;
	sd.pDelta=m_delta;
	sd.pWeight=m_weight;

	pPP->start(d_softmaxProcessThread,&sd,m_curNumFeature);
	pPP->wait();

	for(int i=0;i<m_curNumFeature;i++)
	{
		precision sum=0;
		for(int j=0;j<batch;j++)
			sum+=m_delta[j*m_curNumFeature+i];
		m_biasGrad[i]=sum/batch;
	}

};


double  CSoftmaxLayerCPU::getCost(DLparam& params)
{
	double finSum=getWeightCost();

	double preSum=0.0;
	for(int i=0;i<batch;i++)
		preSum+=log(m_predDatas[i*m_curNumFeature+params.labels[i]]);

	preSum/=-batch;
	return preSum+finSum;
};