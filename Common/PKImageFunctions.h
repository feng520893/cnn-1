#ifndef PKIMAGEFUNCTIONS_H
#define PKIMAGEFUNCTIONS_H
#include "pkDefine.h"
#include "CpkMat.h"

namespace pk
{

	enum ZOOM_TYPE
	{ 
		NEAREST,
		BILINEAR,
	};

	int SaveImage(const char*path,CpkMat& src);
	int loadImage(const char*path,CpkMat& dest);

	//��ȡRGB��ͨ��
	int GetRGBchannel(CpkMat&src,CpkMat& colorR,CpkMat&colorG,CpkMat& colorB);
	//rgbת�Ҷ�
	bool RGBtoGrayscale(CpkMat&dest,CpkMat&src);

	//��ɫ
	int RevColor(CpkMat&dest,CpkMat&src);

	//����
	int zoom(CpkMat&dest,int widthOut,int heighOut,CpkMat&src,ZOOM_TYPE type);

	//���·�ת
	int UpDown(CpkMat&dest,CpkMat&src);

	//���ҷ�ת
	int LeftRight(CpkMat&dest,CpkMat&src);

	//��ָ�������ȡͼƬ����
	int ReadImageRect(CpkMat&dest,int destWidth,const char*data,int x,int y,int srcWidth,int height);
};
#endif