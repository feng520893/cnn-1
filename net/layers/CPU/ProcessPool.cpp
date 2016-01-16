#include "ProcessPool.h"

CProcessPool* CProcessPool::m_pThis=NULL;

unsigned int __stdcall ProcessThread(LPVOID pM)
{
	CProcessPool::ProcessData* pData=(CProcessPool::ProcessData*)pM;

	CProcessPool::UserData us;
	us.threadBuffers=pData->pThis->m_threadBuffers;
	us.threadNum=pData->pThis->m_threadNum;
	while(1)
	{
		WaitForSingleObject(pData->pThis->m_walkUpSem,INFINITE);

		if(WaitForSingleObject(pData->pThis->m_quit,0)==WAIT_OBJECT_0)
			break;
		us.start=pData->pThis->getIndex(us.end,us.threadID);
		us.pUserData=pData->userData;
		HANDLE fin=pData->pThis->getFinEvent();
		pData->pThis->m_pRunThread(&us);
		SetEvent(fin);
	}
	return 0;
}



CProcessPool::CProcessPool(void)
{
	m_threadBuffers=NULL;
	m_curID=0;
}

CProcessPool::~CProcessPool(void)
{
	if(m_threadBuffers)
		delete [] m_threadBuffers;
}

void CProcessPool::init()
{
	SYSTEM_INFO siSysInfo; 
	GetSystemInfo(&siSysInfo);
	m_threadNum=m_number=1;//siSysInfo.dwNumberOfProcessors;
	m_pThread=new HANDLE[m_number];
	m_pFin=new HANDLE[m_number];

	m_threadBuffers=new double[m_number];
	m_threadBuffersLeng=m_number;
	
	m_walkUpSem=CreateSemaphore(NULL,0,m_number,NULL);

	for(int i=0;i<m_number;i++)
		m_pFin[i]=CreateEvent(NULL,TRUE,FALSE,NULL);

	m_quit=CreateEvent(NULL,TRUE,FALSE,NULL);

	InitializeCriticalSection(&m_cs);


	m_processData.pThis=this;

	for(int i=0;i<m_number;i++)
	{
		m_pThread[i]=(HANDLE)_beginthreadex(NULL,NULL,ProcessThread,&m_processData,NULL,NULL);
	}
}

void CProcessPool::start(unsigned int (WINAPI *pthreadAdd)(LPVOID pM),LPVOID pUserData,int num)
{
	int i=0;
	m_processData.userData=pUserData;
	m_pRunThread=pthreadAdd;
	if(num<m_number)
	{
		m_number=num;
		for(i=0;i<num;i++)
		{
			m_quaueIndex.push(i);
			m_quaueIndex.push(i+1);
			m_quaueOK.push(m_pFin[i]);
		}

		for(int i=0;i<num;i++)
			ReleaseSemaphore(m_walkUpSem,1,NULL);
	}
	else
	{
		for(i=0;i<m_number-1;i++)
		{
			m_quaueIndex.push(i*(num/m_number));
			m_quaueIndex.push((i+1)*(num/m_number));
			m_quaueOK.push(m_pFin[i]);
		}
		m_quaueIndex.push(i*(num/m_number));
		m_quaueIndex.push(num);
		m_quaueOK.push(m_pFin[i]);

		for(int i=0;i<m_number;i++)
			ReleaseSemaphore(m_walkUpSem,1,NULL);
	}
}

void CProcessPool::wait()
{
	WaitForMultipleObjects(m_number,m_pFin,TRUE,INFINITE);
	for(int i=0;i<m_number;i++)
		ResetEvent(m_pFin[i]);
	m_number=m_threadNum;
	m_curID=0;
}

void CProcessPool::destroy()
{
	SetEvent(m_quit);
	for(int i=0;i<m_number;i++)
		ReleaseSemaphore(m_walkUpSem,1,NULL);

	WaitForMultipleObjects(m_number,m_pThread,TRUE,INFINITE);
	for(int i=0;i<m_number;i++)
	{
		CloseHandle(m_pThread[m_number]);
		m_pThread[i]=NULL;
		CloseHandle(m_pFin[m_number]);
		m_pFin[i]=NULL;
	}
	if(m_walkUpSem)
		CloseHandle(m_walkUpSem);
	if(m_pThread)
		delete [] m_pThread;
	if(m_pFin)
		delete [] m_pFin;

	DeleteCriticalSection(&m_cs);
}

HANDLE CProcessPool::getFinEvent()
{
	EnterCriticalSection(&m_cs);
	HANDLE tmp=m_quaueOK.front();
	m_quaueOK.pop();
	LeaveCriticalSection(&m_cs);
	return tmp;
}

int CProcessPool::getIndex(int &end,int& threadID)
{
	EnterCriticalSection(&m_cs);
	threadID=m_curID++;
	int tmp=m_quaueIndex.front();
	m_quaueIndex.pop();
	end=m_quaueIndex.front();
	m_quaueIndex.pop();
	LeaveCriticalSection(&m_cs);
	return tmp;
}