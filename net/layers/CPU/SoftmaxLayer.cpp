#include "SoftmaxLayer.h"



int CSoftmaxLayerCPU::setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs)
{
	int inputSize=inputs[0]->size()/inputs[0]->num;;
	m_weightLen=m_param.curNumFeature*inputSize;
	DL_ASSER(m_weightLen!=0);

	outputs[0]->create(inputs[0]->num,m_param.curNumFeature,1,1);

	return CLayerBaseCPU::setup(inputs,outputs);
}


void CSoftmaxLayerCPU::freeMem()
{
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

precision CSoftmaxLayerCPU::feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	int inputSize=bottoms[0]->size()/bottoms[0]->num;

	precision* predDatas=tops[0]->cpuData;

	softmaxData fd;
	fd.pB=m_bias.cpuData;
	fd.pW=m_weight.cpuData;

	fd.pSrc=bottoms[0]->cpuData;
	fd.inputNumFeature=inputSize;
	fd.numFeature=m_param.curNumFeature;
	fd.batch=bottoms[0]->num;
	fd.pDest=predDatas;

	CProcessPool* pPP=CProcessPool::initInstance();
	pPP->start(softmaxConnectProcessThread,&fd,m_param.curNumFeature);
	pPP->wait();

	double * tmpMax=new double[bottoms[0]->num];

	for(int i=0;i<bottoms[0]->num;i++)
	{
		double max=0.0;
		for(int j=0;j<m_param.curNumFeature;j++)
		{
			if(predDatas[i*m_param.curNumFeature+j]>max)
				max=predDatas[i*m_param.curNumFeature+j];
		}
		tmpMax[i]=max;
	}

	//º∆À„∏≈¬ 
	for(int i=0;i<bottoms[0]->num;i++)
	{
		double tSum=0.0;
		for(int j=0;j<m_param.curNumFeature;j++)
		{
			predDatas[i*m_param.curNumFeature+j]-=tmpMax[i];
			predDatas[i*m_param.curNumFeature+j]=exp(predDatas[i*m_param.curNumFeature+j]);
			tSum+=predDatas[i*m_param.curNumFeature+j];
		}
		for(int j=0;j<m_param.curNumFeature;j++)
			predDatas[i*m_param.curNumFeature+j]/=tSum;
	}
	delete [] tmpMax;

	double finSum=getWeightCost();

	double preSum=0.0;
	for(int i=0;i<bottoms[0]->num;i++)
	{
		int label=(int)bottoms[1]->cpuData[i];
		preSum+=log(predDatas[i*m_param.curNumFeature+label]);
	}

	preSum/=-bottoms[0]->num;
	return preSum+finSum;

};

int CSoftmaxLayerCPU::backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms)
{
	int num=tops[0]->num;
	int inputSize=bottoms[0]->size()/bottoms[0]->num;
	memset(tops[0]->cpuDiff,0,sizeof(precision)*tops[0]->size());
	for(int i=0;i<num;i++)
	{
		int label=(int)bottoms[1]->cpuData[i];
		tops[0]->cpuDiff[i*m_param.curNumFeature+label]=1;
	}

	for(int i=0;i<m_param.curNumFeature*num;i++)
		tops[0]->cpuDiff[i]=-(tops[0]->cpuDiff[i]-tops[0]->cpuData[i]);
	
	matrixMul(tops[0]->cpuDiff,num,m_param.curNumFeature,m_weight.cpuData,m_param.curNumFeature,inputSize,bottoms[0]->cpuDiff);
	return this->getGrad(tops,bottoms);
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

int CSoftmaxLayerCPU::getGrad(std::vector<Blob<precision>*>& tops,std::vector<Blob<precision>*>& bottoms)
{
	int num=tops[0]->num;
	int inputSize=bottoms[0]->size()/bottoms[0]->num;

	CProcessPool* pPP=CProcessPool::initInstance();
	softmaxData2 sd;
	sd.lambda=m_param.lambda;
	sd.numImages=num;
	sd.inputNumFeature=inputSize;
	sd.curNumFeature=m_param.curNumFeature;

	sd.pSrc=bottoms[0]->cpuData;
	sd.pWeightGrad=m_weightGrad;
	sd.pDelta=tops[0]->cpuDiff;
	sd.pWeight=m_weight.cpuData;

	pPP->start(d_softmaxProcessThread,&sd,m_param.curNumFeature);
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