#ifndef SHOW_IMG_BASE_DLG
#define SHOW_IMG_BASE_DLG
#include"../../CpkMat.h"
namespace pk
{
	struct IShowImgBaseDlg
	{
		virtual int CreateDlg(const char* name)=0;
		virtual int ShowDlg(CpkMat& img)=0;
		virtual int DestroyDlg()=0;
	};
};

#endif