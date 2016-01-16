#include"relulayer.h"

struct active_t
{
	precision* src;
	precision* dest;
};

unsigned int __stdcall activeReluThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int end=pUData->end;
	precision res=0.0;
	active_t* pActiveT=(active_t*)pUData->pUserData;
	precision* src=pActiveT->src;
	precision* dest=pActiveT->dest;

	for(int tid=pUData->start;tid<end;tid++)
		dest[tid] = max(0.0,src[tid]);
	return 0;
}

precision CReluLayerCPU::feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	CProcessPool* pPP=CProcessPool::initInstance();
	active_t at;
	at.src=bottoms[0]->cpuData;
	at.dest=tops[0]->cpuData;
	pPP->start(activeReluThread,&at,bottoms[0]->size());
	pPP->wait();
	return NET_SUCCESS;
}

unsigned int __stdcall d_activeReluThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int end=pUData->end;
	precision res=0.0;
	active_t* pActiveT=(active_t*)pUData->pUserData;
	precision* src=pActiveT->src;
	precision* dest=pActiveT->dest;

	for(int tid=pUData->start;tid<end;tid++)
	{
		if(src[tid]>0.0)
			dest[tid] = 1.0;
		else
		    dest[tid] = 0.0;
	}
	return 0;
}

int CReluLayerCPU::backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms)
{
	CProcessPool* pPP=CProcessPool::initInstance();
	active_t at;
	at.src=tops[0]->cpuData;
	at.dest=bottoms[0]->cpuData;
	pPP->start(d_activeReluThread,&at,tops[0]->size());
	pPP->wait();
	return NET_SUCCESS;
}