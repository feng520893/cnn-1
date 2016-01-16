#include"dropoutLayer.h"

int CDropoutLayerCPU::setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs)
{
	m_mask.create(inputs[0]->num,inputs[0]->dataChannel,inputs[0]->dimHeight,inputs[0]->dimWidth);
	return NET_SUCCESS;
}

struct bernoulli_t
{
	precision* mask;
	float rate;
};

unsigned int __stdcall bernoulliThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int end=pUData->end;
	precision res=0.0;
	bernoulli_t* pBernoulliT=(bernoulli_t*)pUData->pUserData;
	precision* mask=pBernoulliT->mask;

	for(int tid=pUData->start;tid<end;tid++)
	{
		if(mask[tid]>pBernoulliT->rate)
			mask[tid]=1;
		else
			mask[tid]=0;
	}
	return 0;
}

struct dropout_t
{
	precision* src;
	precision* dest;
	precision* mask;
	float scale;
};

unsigned int __stdcall dropoutThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int end=pUData->end;
	precision res=0.0;
	dropout_t* pDropoutT=(dropout_t*)pUData->pUserData;

	precision* src=pDropoutT->src;
	precision* dest=pDropoutT->dest;
	precision* mask=pDropoutT->mask;

	for(int tid=pUData->start;tid<end;tid++)
		dest[tid] = src[tid] * mask[tid] * pDropoutT->scale;
	return 0;
}

precision CDropoutLayerCPU::feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	if(m_param.phase == TRAIN)
	{
		float threshold_=m_param.dropoutParam.dropoutRate;
		float scale=1;
		if(threshold_!=1)
			scale=1 / (1 - threshold_);

		CTools::cpuRand<>(m_mask.cpuData,m_mask.size(),CTools::NORMAL);

		CProcessPool* pPP=CProcessPool::initInstance();
		bernoulli_t bt;
		bt.mask=m_mask.cpuData;
		bt.rate=1-threshold_;
		pPP->start(bernoulliThread,&bt,m_mask.size());
		pPP->wait();

		dropout_t dt;
		dt.src=bottoms[0]->cpuData;
		dt.dest=tops[0]->cpuData;
		dt.mask=m_mask.cpuData;
		dt.scale=scale;

		pPP->start(dropoutThread,&dt,bottoms[0]->size());
		pPP->wait();
	}
	return NET_SUCCESS;
}

int CDropoutLayerCPU::backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms)
{
	if(m_param.phase == TRAIN)
	{
		CProcessPool* pPP=CProcessPool::initInstance();

		float threshold_=m_param.dropoutParam.dropoutRate;
		float scale=1;
		if(threshold_!=1)
			scale=1 / (1 - threshold_);

		dropout_t dt;
		dt.src=tops[0]->cpuDiff;
		dt.dest=bottoms[0]->cpuDiff;
		dt.mask=m_mask.cpuData;
		dt.scale=scale;

		pPP->start(dropoutThread,&dt,tops[0]->size());
		pPP->wait();
	}
	return NET_SUCCESS;
}