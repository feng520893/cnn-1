#ifndef PKIMAGEFUNCTIONS_H
#define PKIMAGEFUNCTIONS_H
#include "pkDefine.h"
#include"PCA.h"
#include "CpkMat.h"


namespace pk
{

	enum ZOOM_TYPE
	{ 
		NEAREST,
		BILINEAR,
	};

	int imwrite(const char*path,CpkMat& src);
	int imread(const char*path,CpkMat& dest);

	//二值化,threshold=0调用大津自动閥值
	int Binary(CpkMat&src,int threshold=0);

	//中值滤波
	int MedF(double **s,int w,int h,int n);

	//高斯肤色建模
	int GaussSkinModel(CpkMat&destImg,CpkMat&srcImg);

	//膨胀：
	int dilation(CpkMat&dest,CpkMat&src,CpkMat&mask);

	//腐蚀：
	int erosion(CpkMat&dest,CpkMat&src,CpkMat&mask);

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