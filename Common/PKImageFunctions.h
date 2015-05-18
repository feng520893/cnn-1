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

	//��ֵ��,threshold=0���ô���Զ��yֵ
	int Binary(CpkMat&src,int threshold=0);

	//��ֵ�˲�
	int MedF(double **s,int w,int h,int n);

	//��˹��ɫ��ģ
	int GaussSkinModel(CpkMat&destImg,CpkMat&srcImg);

	//���ͣ�
	int dilation(CpkMat&dest,CpkMat&src,CpkMat&mask);

	//��ʴ��
	int erosion(CpkMat&dest,CpkMat&src,CpkMat&mask);

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