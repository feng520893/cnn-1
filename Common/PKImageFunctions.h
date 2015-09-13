#ifndef PKIMAGEFUNCTIONS_H
#define PKIMAGEFUNCTIONS_H
#include "pkDefine.h"
#include"PCA.h"
#include "CpkMat.h"
#include<vector>

namespace pk
{

	enum ZOOM_TYPE
	{ 
		NEAREST=0,
		BILINEAR,
	};

	enum CHANGE_IMAGE_FORMAT
	{
		BGR2HSV=0,
		BGR2GRAY,
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

	//��ȡͨ��(ע��RGB����˳����BGR)
	int Split(CpkMat&src,std::vector<CpkMat>& mats);

	//�ı�ͼƬ��ʽ
	int ChangeImageFormat(CpkMat src,CpkMat&dest,CHANGE_IMAGE_FORMAT);

	//ֱ��ͼ���⻯
	int EqualizeHist(CpkMat& dest,CpkMat& src);

	//��ɫ
	int RevColor(CpkMat&dest,CpkMat&src);

	//����
	int zoom(CpkMat&dest,int widthOut,int heighOut,CpkMat&src,ZOOM_TYPE type);

	//�ȱ������ţ��ٰѽϳ��ı߲ü�
	int zoomMidImage(CpkMat& dd,int zoomSize,ZOOM_TYPE type);

	//���·�ת
	int UpDown(CpkMat&dest,CpkMat&src);

	//���ҷ�ת
	int LeftRight(CpkMat&dest,CpkMat&src);

	//��ָ�������ȡͼƬ����
	int ReadImageRect(CpkMat&dest,int destWidth,const char*data,int x,int y,int srcWidth,int height);
};
#endif