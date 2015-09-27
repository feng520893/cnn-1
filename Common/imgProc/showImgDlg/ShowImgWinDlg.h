#pragma once
#include"ShowImgBaseDlg.h"

#include <Windows.h>

namespace pk
{
	class CShowImgWinDlg : public IShowImgBaseDlg
	{
	public:
		CShowImgWinDlg(void);
		~CShowImgWinDlg(void);
		int CreateDlg(const char* name);
		int ShowDlg(CpkMat& img);
		int DestroyDlg();

		HWND m_hWnd;
		HANDLE m_finEvent;
		char m_name[1024];
		DWORD m_dlgError;

		static unsigned __stdcall DlgProcThread(void * pParam);
		static int m_dlgCount;
	private:
		HANDLE m_handle;

	};
};