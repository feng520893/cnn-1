#include"tanhlayer.h"

struct active_t
{
	precision* src;
	precision* dest;
};

unsigned int __stdcall activeTanhThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int end=pUData->end;
	precision res=0.0;
	active_t* pActiveT=(active_t*)pUData->pUserData;
	precision* src=pActiveT->src;
	precision* dest=pActiveT->dest;

	for(int tid=pUData->start;tid<end;tid++)
		dest[tid] = ::tanh(src[tid] * 2.0 / 3.0) * 1.7159;
	return 0;
}

precision CTanhLayerCPU::feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	CProcessPool* pPP=CProcessPool::initInstance();
	active_t at;
	at.src=bottoms[0]->cpuData;
	at.dest=tops[0]->cpuData;
	pPP->start(activeTanhThread,&at,bottoms[0]->size());
	pPP->wait();
	return NET_SUCCESS;
}

unsigned int __stdcall d_activeTanhThread(LPVOID pM)
{
	CProcessPool::UserData* pUData=(CProcessPool::UserData*)pM;
	int end=pUData->end;
	precision res=0.0;
	active_t* pActiveT=(active_t*)pUData->pUserData;
	precision* src=pActiveT->src;
	precision* dest=pActiveT->dest;

	for(int tid=pUData->start;tid<end;tid++)
	{
		precision res = 1.7159; 
		precision temp = src[tid] * src[tid] / 1.7159; 
		dest[tid] = (res - temp) * 2.0 / 3.0; 
	}
	return 0;
}

int CTanhLayerCPU::backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms)
{
	CProcessPool* pPP=CProcessPool::initInstance();
	active_t at;
	at.src=tops[0]->cpuData;
	at.dest=bottoms[0]->cpuData;
	pPP->start(d_activeTanhThread,&at,tops[0]->size());
	pPP->wait();
	return NET_SUCCESS;
}