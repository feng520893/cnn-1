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

	//获取RGB三通道
	int GetRGBchannel(CpkMat&src,CpkMat& colorR,CpkMat&colorG,CpkMat& colorB);
	//rgb转灰度
	bool RGBtoGrayscale(CpkMat&dest,CpkMat&src);

	//反色
	int RevColor(CpkMat&dest,CpkMat&src);

	//缩放
	int zoom(CpkMat&dest,int widthOut,int heighOut,CpkMat&src,ZOOM_TYPE type);

	//上下翻转
	int UpDown(CpkMat&dest,CpkMat&src);

	//左右翻转
	int LeftRight(CpkMat&dest,CpkMat&src);

	//按指定区域获取图片数据
	int ReadImageRect(CpkMat&dest,int destWidth,const char*data,int x,int y,int srcWidth,int height);
};
#endif