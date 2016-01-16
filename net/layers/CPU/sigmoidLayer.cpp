#include"sigmoidlayer.h"

struct active_t
{
	precision* src;
	precision* dest;
};

unsigned int __stdcall activeSigmoidThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int end=pUData->end;
	precision res=0.0;
	active_t* pActiveT=(active_t*)pUData->pUserData;
	precision* src=pActiveT->src;
	precision* dest=pActiveT->dest;

	for(int tid=pUData->start;tid<end;tid++)
		dest[tid]=1/(1+::exp(-src[tid]));
	return 0;
}

precision CSigmoidLayerCPU::feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	CProcessPool* pPP=CProcessPool::initInstance();
	active_t at;
	at.src=bottoms[0]->cpuData;
	at.dest=tops[0]->cpuData;
	pPP->start(activeSigmoidThread,&at,bottoms[0]->size());
	pPP->wait();
	return NET_SUCCESS;
}

unsigned int __stdcall d_activeSigmoidThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int end=pUData->end;
	precision res=0.0;
	active_t* pActiveT=(active_t*)pUData->pUserData;
	precision* src=pActiveT->src;
	precision* dest=pActiveT->dest;

	for(int tid=pUData->start;tid<end;tid++)
		dest[tid]=src[tid]*(1-src[tid]); 
	return 0;
}

int CSigmoidLayerCPU::backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms)
{
	CProcessPool* pPP=CProcessPool::initInstance();
	active_t at;
	at.src=tops[0]->cpuData;
	at.dest=bottoms[0]->cpuData;
	pPP->start(d_activeSigmoidThread,&at,tops[0]->size());
	pPP->wait();
	return NET_SUCCESS;
}