#include "LayerBaseCPU.h"

int CLayerBaseCPU::initMem()
{
	m_vecBias=new precision[m_curNumFeature];
	DL_ASSER(m_vecBias);

	memset(m_vecBias,0,sizeof(precision)*m_curNumFeature);
	
	m_bias=new precision[m_curNumFeature];
	DL_ASSER(m_bias);
	memset(m_bias,0,sizeof(precision)*m_curNumFeature);

	m_biasGrad=new precision[m_curNumFeature];
	DL_ASSER(m_biasGrad);

	m_weight=new precision[m_weightLen];
	DL_ASSER(m_weight);
	pk::randn(m_weight,m_weightLen);

	m_weightGrad=new precision[m_weightLen];
	DL_ASSER(m_weightGrad);

	m_vecWeight=new precision[m_weightLen];
	DL_ASSER(m_vecWeight);
	memset(m_vecWeight,0,sizeof(precision)*m_weightLen);

	return PK_SUCCESS;
}

void CLayerBaseCPU::freeMem()
{
	CPU_FREE(m_vecBias);
	CPU_FREE(m_vecWeight);
	CPU_FREE(m_weight);
	CPU_FREE(m_bias);
	CPU_FREE(m_weightGrad);
	CPU_FREE(m_biasGrad);
	CPU_FREE(m_delta);
}

void CLayerBaseCPU::setWeightValue(int index,double value)
{
	m_weight[index]=value;
}

double CLayerBaseCPU::getWeightValue(int index)
{
	return m_weight[index];
}

void CLayerBaseCPU::setBiasValue(int index,double value)
{
	m_bias[index]=value;
}

double CLayerBaseCPU::getBiasValue(int index)
{
	return m_bias[index];
}

void CLayerBaseCPU::getWeightsGrad(double* gradWeightData)
{
	memcpy(gradWeightData,m_weightGrad,sizeof(double)*m_weightLen);
}

void CLayerBaseCPU::getBiasGrad(double* gradBiasData)
{
	memcpy(gradBiasData,m_biasGrad,sizeof(double)*m_curNumFeature);
}

int CLayerBaseCPU::save(FILE* fp)
{
	precision* pTmp=new precision[m_weightLen+m_curNumFeature];
	DL_ASSER(pTmp);

	memcpy(pTmp,m_weight,sizeof(precision)*m_weightLen);

	memcpy(pTmp+m_weightLen,m_bias,sizeof(precision)*m_curNumFeature);

	fwrite(pTmp,sizeof(precision)*(m_weightLen+m_curNumFeature),1,fp);

	memcpy(pTmp,m_vecWeight,sizeof(precision)*m_weightLen);

	memcpy(pTmp+m_weightLen,m_vecBias,sizeof(precision)*m_curNumFeature);

	fwrite(pTmp,sizeof(precision)*(m_weightLen+m_curNumFeature),1,fp);

	delete [] pTmp;
	return PK_SUCCESS;
}

int CLayerBaseCPU::load(FILE*fp)
{
	precision* pTmp=new precision[m_weightLen+m_curNumFeature];
	DL_ASSER(pTmp!=NULL);

	fread(pTmp,sizeof(precision)*(m_weightLen+m_curNumFeature),1,fp);
	memcpy(m_weight,pTmp,sizeof(precision)*m_weightLen);

	memcpy(m_bias,pTmp+m_weightLen,sizeof(precision)*m_curNumFeature);

	fread(pTmp,sizeof(precision)*(m_weightLen+m_curNumFeature),1,fp);
	memcpy(m_vecWeight,pTmp,sizeof(precision)*m_weightLen);

	memcpy(m_vecBias,pTmp+m_weightLen,sizeof(precision)*m_curNumFeature);

	delete [] pTmp;
	return PK_SUCCESS;
}

void CLayerBaseCPU::updateWeight(float mom,float alpha)
{
	if(m_weightLen==0)
		return;
	for(int i=0;i<m_weightLen;i++)
	{
		m_vecWeight[i]=m_vecWeight[i]*mom+m_weightGrad[i]*alpha;
		m_weight[i]=m_weight[i]-m_vecWeight[i];
	}

	for(int i=0;i<m_curNumFeature;i++)
	{
		m_vecBias[i]=m_vecBias[i]*mom+m_biasGrad[i]*alpha;
		m_bias[i]=m_bias[i]-m_vecBias[i];
	}
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
	return PK_SUCCESS;
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
	pPP->start(GetWeightsCostThread,m_weight,m_weightLen);
	pPP->wait();
	double res=0.0;
	for(int i=0;i<pPP->m_threadNum;i++)
		res+=pPP->m_threadBuffers[i];
	return res*m_lambda/2;
}

double activeFunCPU(double src,int type)
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
	else
	{
		return 1/(1+::exp(-src));
	}

	return 0.0;
}

double d_activeFunCPU(double src,int type)
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
	else
	{
		return src*(1-src);
	}
	return 0;
}

unsigned int __stdcall NolineProcessThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int end=pUData->end;
	nolineData* pUserData=(nolineData*)pUData->pUserData;
	for(int tid=pUData->start;tid<end;tid++)
		pUserData->deltaE[tid]=pUserData->deltaS[tid]*d_activeFunCPU(pUserData->activeData[tid],pUserData->type);
	return 0;
}