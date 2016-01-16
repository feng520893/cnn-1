#pragma once
#include <Windows.h>
#include<process.h>
#include <queue>

class CProcessPool
{
public:
	
	HANDLE  m_quit;
	HANDLE  m_walkUpSem;

	double* m_threadBuffers;
	unsigned int m_threadBuffersLeng;
	int m_threadNum;

	unsigned int (WINAPI *m_pRunThread)(LPVOID pData);

	struct UserData
	{
		int start;
		int end;
		double* threadBuffers;
		int threadID;
		int threadNum;
		LPVOID pUserData;
	};

	struct ProcessData
	{
		CProcessPool* pThis;
		LPVOID userData;
	};

	static CProcessPool* initInstance()
	{
		if(m_pThis==NULL)
		{
			m_pThis=new CProcessPool;
			m_pThis->init();
		}
		return m_pThis;
	}

	void start(unsigned int (WINAPI *pthreadAdd)(LPVOID pM),LPVOID pUserData,int num);
	void wait();
	void destroy();
	HANDLE getFinEvent();
	int    getIndex(int &end,int& threadID);
	virtual ~CProcessPool(void);
private:

	CProcessPool(void);
	void init();

	HANDLE* m_pThread;
	HANDLE* m_pFin;
	
	std::queue<HANDLE> m_quaueOK;
	std::queue<int>    m_quaueIndex;
	int m_number;
	int m_curID;
	ProcessData m_processData;
	CRITICAL_SECTION m_cs;

	static CProcessPool* m_pThis;
};