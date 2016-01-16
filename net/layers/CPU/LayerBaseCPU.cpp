#include "LayerBaseCPU.h"

int CLayerBaseCPU::setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs)
{
	m_vecBias=new precision[m_param.curNumFeature];
	DL_ASSER(m_vecBias);

	memset(m_vecBias,0,sizeof(precision)*m_param.curNumFeature);
	
	m_bias.create(1,1,1,m_param.curNumFeature);
	memset(m_bias.cpuData,0,sizeof(precision)*m_param.curNumFeature);

	m_biasGrad=new precision[m_param.curNumFeature];
	DL_ASSER(m_biasGrad);

	m_weight.create(1,1,1,m_weightLen);
	CTools::cpuRand<precision>(m_weight.cpuData,m_weightLen,CTools::GAUSS);

	m_weightGrad=new precision[m_weightLen];
	DL_ASSER(m_weightGrad);

	m_vecWeight=new precision[m_weightLen];
	DL_ASSER(m_vecWeight);
	memset(m_vecWeight,0,sizeof(precision)*m_weightLen);

	return CLayer::setup(inputs,outputs);
}

void CLayerBaseCPU::freeMem()
{
	CPU_FREE(m_vecBias);
	CPU_FREE(m_vecWeight);
	CPU_FREE(m_weightGrad);
	CPU_FREE(m_biasGrad);
}

void CLayerBaseCPU::setWeightValue(int index,double value)
{
	m_weight.cpuData[index]=value;
}

double CLayerBaseCPU::getWeightValue(int index)
{
	return m_weight.cpuData[index];
}

void CLayerBaseCPU::setBiasValue(int index,double value)
{
	m_bias.cpuData[index]=value;
}

double CLayerBaseCPU::getBiasValue(int index)
{
	return m_bias.cpuData[index];
}

void CLayerBaseCPU::getWeightsGrad(double* gradWeightData)
{
	memcpy(gradWeightData,m_weightGrad,sizeof(double)*m_weightLen);
}

void CLayerBaseCPU::getBiasGrad(double* gradBiasData)
{
	memcpy(gradBiasData,m_biasGrad,sizeof(double)*m_param.curNumFeature);
}

int CLayerBaseCPU::save(FILE* fp)
{
	precision* pTmp=new precision[m_weightLen+m_param.curNumFeature];
	DL_ASSER(pTmp);

	memcpy(pTmp,m_weight.cpuData,sizeof(precision)*m_weightLen);

	memcpy(pTmp+m_weightLen,m_bias.cpuData,sizeof(precision)*m_param.curNumFeature);

	fwrite(pTmp,sizeof(precision)*(m_weightLen+m_param.curNumFeature),1,fp);

	memcpy(pTmp,m_vecWeight,sizeof(precision)*m_weightLen);

	memcpy(pTmp+m_weightLen,m_vecBias,sizeof(precision)*m_param.curNumFeature);

	fwrite(pTmp,sizeof(precision)*(m_weightLen+m_param.curNumFeature),1,fp);

	delete [] pTmp;
	return NET_SUCCESS;
}

int CLayerBaseCPU::load(FILE*fp)
{
	precision* pTmp=new precision[m_weightLen+m_param.curNumFeature];
	DL_ASSER(pTmp!=NULL);

	fread(pTmp,sizeof(precision)*(m_weightLen+m_param.curNumFeature),1,fp);
	memcpy(m_weight.cpuData,pTmp,sizeof(precision)*m_weightLen);

	memcpy(m_bias.cpuData,pTmp+m_weightLen,sizeof(precision)*m_param.curNumFeature);

	fread(pTmp,sizeof(precision)*(m_weightLen+m_param.curNumFeature),1,fp);
	memcpy(m_vecWeight,pTmp,sizeof(precision)*m_weightLen);

	memcpy(m_vecBias,pTmp+m_weightLen,sizeof(precision)*m_param.curNumFeature);

	delete [] pTmp;
	return NET_SUCCESS;
}

int CLayerBaseCPU::updateWeight(float mom,float baseLR)
{
	if(m_weightLen == 0)
		return NET_SUCCESS;

	float biasLR=baseLR*m_param.biasLearnRatio;
	float weightsLR=baseLR*m_param.weightLearnRatio;

	for(int i=0;i<m_weightLen;i++)
	{
		m_vecWeight[i]=m_vecWeight[i]*mom+m_weightGrad[i]*weightsLR;
		m_weight.cpuData[i]=m_weight.cpuData[i]-m_vecWeight[i];
	}

	for(int i=0;i<m_param.curNumFeature;i++)
	{
		m_vecBias[i]=m_vecBias[i]*mom+m_biasGrad[i]*biasLR;
		m_bias.cpuData[i]=m_bias.cpuData[i]-m_vecBias[i];
	}
	return NET_SUCCESS;
}

int CLayerBaseCPU::matrixMul(precision* x,int rowL,int colL,precision* y,int rowR,int colR,precision* z)
{
	double* pX1=x;
	double* pX2=y;
	double* pDest=z;
	for (int i = 0; i <rowL; i++)   
	{   
		for(int k=0;k<colR;k++)
		{
			precision sum=0.0;
			for (int j = 0; j <colL; j++)   
			{   
				sum+= pX1[i*colL+j]*pX2[j*colR+k];   				
			}   
			pDest[i*colR+k]=sum;	
		}
	}   
	return NET_SUCCESS;
}

unsigned int __stdcall GetWeightsCostThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int end=pUData->end;
	precision res=0.0;
	precision* pWeights=(precision*)pUData->pUserData;

	for(int tid=pUData->start;tid<end;tid++)
		res+=::pow(pWeights[tid],2);
	pUData->threadBuffers[pUData->threadID]=res;
	return 0;
}

double CLayerBaseCPU::getWeightCost()
{
	CProcessPool* pPP=CProcessPool::initInstance();
	pPP->start(GetWeightsCostThread,m_weight.cpuData,m_weightLen);
	pPP->wait();
	double res=0.0;
	for(int i=0;i<pPP->m_threadNum;i++)
		res+=pPP->m_threadBuffers[i];
	return res*m_param.lambda/2;
}

unsigned int __stdcall NolineProcessThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int end=pUData->end;
	nolineData* pUserData=(nolineData*)pUData->pUserData;
	for(int tid=pUData->start;tid<end;tid++)
		pUserData->deltaE[tid]=pUserData->deltaS[tid]*pUserData->activeData[tid];
	return 0;
}