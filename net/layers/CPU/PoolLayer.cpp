#include "PoolLayer.h"


int CPoolLayerCPU::setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs)
{

	PoolParam poolParam=m_param.poolParam;
	int dataHeightDim=ceil((float)inputs[0]->dimHeight/poolParam.kernelDim);
	int dataWidthDim=ceil((float)inputs[0]->dimWidth/poolParam.kernelDim);
	outputs[0]->create(inputs[0]->num,inputs[0]->dataChannel,dataHeightDim,dataWidthDim);
	m_maxIndexData.create(inputs[0]->num,inputs[0]->dataChannel,dataHeightDim,dataWidthDim);

	return NET_SUCCESS;
}

struct poolData
{
	precision* pSrc;
	precision* pPoolData;
	precision* pMaxIndex;

	int kernelDim;
	int convDim;
	int numFeature;
	int batch;
	int poolDim;
	int preDim;
};

unsigned int __stdcall MaxPoolThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	poolData* pData=(poolData*)pUData->pUserData;
	precision* srcData=pData->pSrc;
	precision* poolData=pData->pPoolData;
	precision* maxIndex=pData->pMaxIndex;
	int batch=pData->batch;
	int numFeature=pData->numFeature;
	int kernelDim=pData->kernelDim;
	int kernelDim2=kernelDim*kernelDim;
	int poolDim=pData->poolDim;
	int preDim=pData->preDim;
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
		for(int y=0;y<kernelDim&&convY+y<preDim;y++)
		{
			for(int x=0;x<kernelDim&&convX+x<preDim;x++)
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
	int preDim=pData->preDim;
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
		for(int y=0;y<kernelDim&&convY+y<preDim;y++)
			for(int x=0;x<kernelDim&&convX+x<preDim;x++)
				sum+=preFeatureData[(convY+y)*preDim+convX+x];
		poolData[tid]=sum/kernelDim2;
	}
	return 0;
}

precision CPoolLayerCPU::feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	CProcessPool* pPP=CProcessPool::initInstance();
	poolData mp;
	mp.convDim=bottoms[0]->dimWidth;
	mp.kernelDim=m_param.poolParam.kernelDim;
	mp.numFeature=bottoms[0]->dataChannel;
	mp.pMaxIndex=m_maxIndexData.cpuData;
	mp.pPoolData=tops[0]->cpuData;
	mp.pSrc=bottoms[0]->cpuData;
	mp.batch=bottoms[0]->num;
	mp.poolDim=tops[0]->dimWidth;
	mp.preDim=bottoms[0]->dimWidth;

	int poolDataLeng=mp.poolDim*mp.poolDim*mp.batch*mp.numFeature;
	if(m_param.poolParam.poolType==MAX_POOL)
		pPP->start(MaxPoolThread,&mp,poolDataLeng);
	else
		pPP->start(AvgPoolThread,&mp,poolDataLeng);
	pPP->wait();

	return 0;
};

struct poolBackData
{
	precision* poolDelta;
	precision* preDelta;
	precision* maxIndex;
	int batch;
	int numFeature;
	int kernelDim;
	int poolDim;
	int preDim;
};

unsigned int __stdcall MaxPoolBackpropagationThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int row=pUData->start;
	int end=pUData->end;
	poolBackData* pData=(poolBackData*)pUData->pUserData;

	precision* poolDelta=pData->poolDelta;
	precision* preDelta=pData->preDelta;
	precision* maxIndex=pData->maxIndex;
	int batch=pData->batch;
	int numFeature=pData->numFeature;
	int kernelDim=pData->kernelDim;
	int poolDim=pData->poolDim;
	int preDim=pData->preDim;
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
	int preDim=pData->preDim;
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

int CPoolLayerCPU::backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms)
{
	int num=tops[0]->num;
	int inputSize=bottoms[0]->size()/bottoms[0]->num;

	memset(bottoms[0]->cpuDiff,0,sizeof(precision)*bottoms[0]->size());

	poolBackData mpbd;
	mpbd.batch=num;
	mpbd.numFeature=bottoms[0]->dataChannel;
	mpbd.kernelDim=m_param.poolParam.kernelDim;
	mpbd.poolDim=tops[0]->dimWidth;
	mpbd.preDim =bottoms[0]->dimWidth;
	mpbd.maxIndex=m_maxIndexData.cpuData;
	mpbd.preDelta=bottoms[0]->cpuDiff;
	mpbd.poolDelta=tops[0]->cpuDiff;
	
	CProcessPool* pPP=CProcessPool::initInstance();
	int poolDataLeng=num*mpbd.numFeature*tops[0]->dimWidth*tops[0]->dimWidth;
	if(m_param.poolParam.poolType==MAX_POOL)
		pPP->start(MaxPoolBackpropagationThread,&mpbd,poolDataLeng);
	else
		pPP->start(AvgPoolBackpropagationThread,&mpbd,poolDataLeng);
	pPP->wait();
	return 0;
};