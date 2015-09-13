#include "PoolLayerCPU.h"


int CPoolLayerCPU::initMem()
{
	DL_ASSER(m_preConvDim != 0);

	unsigned const int poolDataSize=m_preConvDim*m_preConvDim*batch*m_curNumFeature/m_kernelDim/m_kernelDim;

	m_delta=new precision[poolDataSize];
	DL_ASSER(m_delta);

	m_poolData=new precision[poolDataSize];
	DL_ASSER(m_poolData);

	m_maxIndexData=new int[poolDataSize];
	DL_ASSER(m_maxIndexData);

	return PK_SUCCESS;
}


void CPoolLayerCPU::freeMem()
{
	CPU_FREE(m_maxIndexData);
	CPU_FREE(m_poolData);
	CLayerBaseCPU::freeMem();
}

struct poolData
{
	precision* pSrc;
	precision* pPoolData;
	int*       pMaxIndex;

	int kernelDim;
	int convDim;
	int numFeature;
	int batch;
	int poolDim;
};

unsigned int __stdcall MaxPoolThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	poolData* pData=(poolData*)pUData->pUserData;
	precision* srcData=pData->pSrc;
	precision* poolData=pData->pPoolData;
	int*       maxIndex=pData->pMaxIndex;
	int batch=pData->batch;
	int numFeature=pData->numFeature;
	int kernelDim=pData->kernelDim;
	int kernelDim2=kernelDim*kernelDim;
	int poolDim=pData->poolDim;
	int preDim=poolDim*kernelDim;
	int featureLeng=poolDim*poolDim;
	int batchLeng=featureLeng*pData->numFeature;

	int preFeatureLeng=preDim*preDim;
	int preBatchLeng=preFeatureLeng*pData->numFeature;

	int curIndex=0;
	for(int tid=pUData->start;tid<pUData->end;tid++)
	{
		int batchNo=tid/batchLeng;
		int featureNo=tid%batchLeng;
		curIndex=featureNo%featureLeng;
		featureNo/=featureLeng;

		int convX=(curIndex%poolDim)*kernelDim;
		int convY=(curIndex/poolDim)*kernelDim;

		precision* preFeatureData=srcData+batchNo*preBatchLeng+featureNo*preFeatureLeng;
		precision max=0.0;
		int       index=0;
		for(int y=0;y<kernelDim;y++)
		{
			for(int x=0;x<kernelDim;x++)
			{
				precision tmp=preFeatureData[(convY+y)*preDim+convX+x];
				if(tmp>max)
				{
					max=tmp;
					index=y*kernelDim+x;
				}
			}
		}
		maxIndex[tid]=index;
		poolData[tid]=max;
	}
	return 0;
}

unsigned int __stdcall AvgPoolThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	poolData* pData=(poolData*)pUData->pUserData;
	precision* srcData=pData->pSrc;
	precision* poolData=pData->pPoolData;
	int batch=pData->batch;
	int numFeature=pData->numFeature;
	int kernelDim=pData->kernelDim;
	int kernelDim2=kernelDim*kernelDim;
	int poolDim=pData->poolDim;
	int preDim=poolDim*kernelDim;
	int featureLeng=poolDim*poolDim;
	int batchLeng=featureLeng*pData->numFeature;

	int preFeatureLeng=preDim*preDim;
	int preBatchLeng=preFeatureLeng*pData->numFeature;

	int curIndex=0;
	for(int tid=pUData->start;tid<pUData->end;tid++)
	{
		int batchNo=tid/batchLeng;
		int featureNo=tid%batchLeng;
		curIndex=featureNo%featureLeng;
		featureNo/=featureLeng;

		int convX=(curIndex%poolDim)*kernelDim;
		int convY=(curIndex/poolDim)*kernelDim;

		precision* preFeatureData=srcData+batchNo*preBatchLeng+featureNo*preFeatureLeng;
		precision sum=0;
		for(int y=0;y<kernelDim;y++)
			for(int x=0;x<kernelDim;x++)
				sum+=preFeatureData[(convY+y)*preDim+convX+x];
		poolData[tid]=sum/kernelDim2;
	}
	return 0;
}

void CPoolLayerCPU::feedforward(double* srcData,DLparam& params)
{
	CProcessPool* pPP=CProcessPool::initInstance();
	poolData mp;
	mp.convDim=m_preConvDim;
	mp.kernelDim=m_kernelDim;
	mp.numFeature=m_curNumFeature;
	mp.pMaxIndex=m_maxIndexData;
	mp.pPoolData=m_poolData;
	mp.pSrc=srcData;
	mp.batch=batch;
	mp.poolDim=m_preConvDim/m_kernelDim;

	int poolDataLeng=mp.poolDim*mp.poolDim*batch*m_curNumFeature;
	if(m_poolType==MAX_POOL)
		pPP->start(MaxPoolThread,&mp,poolDataLeng);
	else
		pPP->start(AvgPoolThread,&mp,poolDataLeng);
	pPP->wait();
};

struct poolBackData
{
	precision* poolDelta;
	precision* preDelta;
	int*       maxIndex;
	int batch;
	int numFeature;
	int kernelDim;
	int poolDim;
};

unsigned int __stdcall MaxPoolBackpropagationThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int row=pUData->start;
	int end=pUData->end;
	poolBackData* pData=(poolBackData*)pUData->pUserData;

	precision* poolDelta=pData->poolDelta;
	precision* preDelta=pData->preDelta;
	int*       maxIndex=pData->maxIndex;
	int batch=pData->batch;
	int numFeature=pData->numFeature;
	int kernelDim=pData->kernelDim;
	int poolDim=pData->poolDim;
	int preDim=poolDim*kernelDim;
	int featureLeng=poolDim*poolDim;
	int batchLeng=featureLeng*pData->numFeature;

	int preFeatureLeng=preDim*preDim;
	int preBatchLeng=preFeatureLeng*pData->numFeature;

	int curIndex=0;
	for(;row<end;row++)
	{
		int batchNo=row/batchLeng;
		int featureNo=row%batchLeng;
		curIndex=featureNo%featureLeng;
		featureNo/=featureLeng;

		int convX=(curIndex%poolDim)*kernelDim;
		int convY=(curIndex/poolDim)*kernelDim;

		int index=maxIndex[row];
		int y=index/kernelDim;
		int x=index%kernelDim;
		precision* preFeatureData=preDelta+batchNo*preBatchLeng+featureNo*preFeatureLeng;
		preFeatureData[(convY+y)*preDim+convX+x]=poolDelta[row];
	}
	return 0;
}

unsigned int __stdcall AvgPoolBackpropagationThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	
	poolBackData* pData=(poolBackData*)pUData->pUserData;

	precision* poolDelta=pData->poolDelta;
	precision* preDelta=pData->preDelta;
	int batch=pData->batch;
	int numFeature=pData->numFeature;
	int kernelDim=pData->kernelDim;
	int kernelDim2=kernelDim*kernelDim;
	int poolDim=pData->poolDim;
	int preDim=poolDim*kernelDim;
	int featureLeng=poolDim*poolDim;
	int batchLeng=featureLeng*pData->numFeature;

	int preFeatureLeng=preDim*preDim;
	int preBatchLeng=preFeatureLeng*pData->numFeature;

	int curIndex=0;
	for(int tid=pUData->start;tid<pUData->end;tid++)
	{
		int batchNo=tid/batchLeng;
		int featureNo=tid%batchLeng;
		curIndex=featureNo%featureLeng;
		featureNo/=featureLeng;

		int convX=(curIndex%poolDim)*kernelDim;
		int convY=(curIndex/poolDim)*kernelDim;

		precision* preFeatureData=preDelta+batchNo*preBatchLeng+featureNo*preFeatureLeng;
		for(int y=0;y<kernelDim;y++)
			for(int x=0;x<kernelDim;x++)
				preFeatureData[(convY+y)*preDim+convX+x]=poolDelta[tid]/kernelDim2;
	}
	return 0;
}

void CPoolLayerCPU::backpropagation(double*preDelta,DLparam& params)
{
	memset(preDelta,0,sizeof(precision)*batch*m_curNumFeature*m_preConvDim*m_preConvDim);

	int poolDim=m_preConvDim/m_kernelDim;

	poolBackData mpbd;
	mpbd.batch=batch;
	mpbd.numFeature=m_curNumFeature;
	mpbd.kernelDim=m_kernelDim;
	mpbd.poolDim=poolDim;
	mpbd.maxIndex=m_maxIndexData;
	mpbd.preDelta=preDelta;
	mpbd.poolDelta=m_delta;
	
	CProcessPool* pPP=CProcessPool::initInstance();
	int poolDataLeng=batch*m_curNumFeature*poolDim*poolDim;
	if(m_poolType==MAX_POOL)
		pPP->start(MaxPoolBackpropagationThread,&mpbd,poolDataLeng);
	else
		pPP->start(AvgPoolBackpropagationThread,&mpbd,poolDataLeng);
	pPP->wait();
};