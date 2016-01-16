#include "ShowImgWinDlg.h"

#include <process.h>
#include<cstdio>
namespace pk
{

	int CShowImgWinDlg::m_dlgCount=0;

	LRESULT CALLBACK MainWinProc(HWND hWnd,UINT uMsg,WPARAM wParam,LPARAM lParam)
	{
		switch(uMsg)
		{
		case WM_CREATE:
			{}
			break;
		case WM_PAINT:
			{
				HDC hDC;
				PAINTSTRUCT ps;
				hDC=BeginPaint(hWnd,&ps);
				EndPaint(hWnd,&ps);
			}
			break;
		case WM_DESTROY:
			PostQuitMessage(0);
			break;
		default:
			return DefWindowProc(hWnd,uMsg,wParam,lParam);
		}
		return 0;
	}

	unsigned __stdcall CShowImgWinDlg::DlgProcThread(void * pParam)
	{
		CShowImgWinDlg* pThis=(CShowImgWinDlg*)pParam;

		TCHAR className[1024]={0};
		sprintf(className,"%d",pThis->m_dlgCount++);

		WNDCLASS wnd;
		HINSTANCE hInstance=GetModuleHandle(NULL);
		wnd.style=CS_HREDRAW|CS_VREDRAW;
		wnd.lpfnWndProc=MainWinProc;
		wnd.cbClsExtra=0;
		wnd.cbWndExtra=0;
		wnd.hbrBackground=(HBRUSH)GetStockObject(WHITE_BRUSH);
		wnd.hCursor=NULL;
		wnd.hIcon=NULL;
		wnd.hInstance=hInstance;
		wnd.lpszClassName=className;
		wnd.lpszMenuName=NULL;
		RegisterClass(&wnd);
		int x=0,y=0,width=100,height=100;
		pThis->m_hWnd=CreateWindow(className,pThis->m_name,WS_SYSMENU,x,y,width,height,NULL,(HMENU)0,hInstance,NULL);

		if(pThis->m_hWnd==NULL)
		{
			pThis->m_dlgError=GetLastError();
			return 0;
		}

		UpdateWindow(pThis->m_hWnd);

		SetEvent(pThis->m_finEvent);

		MSG msg;
		while(GetMessage(&msg,NULL,0,0))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		return 0;
	}


	CShowImgWinDlg::CShowImgWinDlg(void)
	{
		m_handle=m_finEvent=m_hWnd=NULL;
	}

	CShowImgWinDlg::~CShowImgWinDlg(void)
	{
		DestroyDlg();
	}

	int CShowImgWinDlg::CreateDlg(const TCHAR* name)
	{
		if(name==NULL)
			return -1;
		sprintf(m_name,"%s",name);

		m_finEvent=CreateEvent(NULL,TRUE,FALSE,NULL);

		m_handle=(HANDLE)_beginthreadex( NULL, 0,&CShowImgWinDlg::DlgProcThread,this,NULL,NULL);
		if(m_handle==NULL)
			return GetLastError();

		WaitForSingleObject(m_finEvent,INFINITE);
		ResetEvent(m_finEvent);

		if(m_dlgError!=0)
			return m_dlgError;
		return 0;
	}

	int CShowImgWinDlg::ShowDlg(CpkMat& img)
	{
		if(!SetWindowPos(m_hWnd,HWND_TOP,0,0,img.Col,img.Row+30,SWP_SHOWWINDOW|SWP_NOMOVE))
			return GetLastError();

		BYTE* tmp=NULL;
		
		if(img.Depth == 1)
		{
			tmp=new BYTE[sizeof(BITMAPINFOHEADER)+sizeof(RGBQUAD)*256];
			memset(tmp,0,sizeof(BITMAPINFOHEADER)+sizeof(RGBQUAD)*256);
		}
		else
		{
			tmp=new BYTE[sizeof(BITMAPINFOHEADER)];
			memset(tmp,0,sizeof(BITMAPINFOHEADER));
		}
		BITMAPINFO* pInfo=(BITMAPINFO*)tmp;
		pInfo->bmiHeader.biWidth=img.Col;	//信息头
		pInfo->bmiHeader.biHeight=img.Row;
		pInfo->bmiHeader.biSize=40;
		pInfo->bmiHeader.biBitCount=img.Depth*8;
		pInfo->bmiHeader.biPlanes=1;

		if(img.Depth == 1 )
		{
			for(int i=0;i<256;i++)
			{
				pInfo->bmiColors[i].rgbBlue=i;	//颜色表数组
				pInfo->bmiColors[i].rgbGreen=i;
				pInfo->bmiColors[i].rgbRed=i;
				pInfo->bmiColors[i].rgbReserved=0;
			}
		}

		HDC hDC=::GetDC(m_hWnd);
		int nScanf=StretchDIBits(hDC,0,0,img.Col,img.Row,0,0,pInfo->bmiHeader.biWidth,pInfo->bmiHeader.biHeight,img.GetData<BYTE>(),pInfo,DIB_RGB_COLORS,SRCCOPY);
		ReleaseDC(m_hWnd,hDC);

		if(tmp)
			delete [] tmp;

		if(nScanf!=img.Row)
			return GetLastError();

		return PK_SUCCESS;
	}

	int CShowImgWinDlg::DestroyDlg()
	{
		if(m_hWnd&&m_handle)
		{
			SendMessage(m_hWnd,WM_DESTROY,NULL,NULL);
			WaitForSingleObject(m_handle,INFINITE);
			CloseHandle(m_handle);
		}
		
		if(m_finEvent)
			CloseHandle(m_finEvent);
		return 0;
	}

};