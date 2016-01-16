#ifndef PKIMAGEFUNCTIONS_H
#define PKIMAGEFUNCTIONS_H
#include "../pkDefine.h"
#include "../CpkMat.h"
#include<vector>

namespace pk
{
	enum ZOOM_TYPE
	{ 
		NEAREST=0,//���ٽ���ֵ�㷨
		BILINEAR, //˫�����ڲ�ֵ�㷨
		CUBIC,    //������ֵ
	};

	enum CHANGE_IMAGE_FORMAT
	{
		BGR2HSV=0,
		BGR2GRAY,
	};

	enum IMAGE_NOISE
	{
		 GAUSSIAN=0,
		 SALT_PEPPER,
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

	//ͼƬ��ʾ
	int createDlg(const char* name);

	int showDlg(const char* name,CpkMat& img);

	int destroyDlg(const char* name);

	//ͼƬ��Ⱦ
	int imNoise(CpkMat& src,CpkMat& dest,IMAGE_NOISE type,double param1=0,double param2=0);

	//��ָ�������ȡͼƬ����
	int ReadImageRect(CpkMat&dest,int destWidth,const char*data,int x,int y,int srcWidth,int height);
};
#endif