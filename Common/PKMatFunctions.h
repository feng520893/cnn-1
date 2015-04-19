#ifndef PK_MAT_FUNCTIONS_H 
#define PK_MAT_FUNCTIONS_H 
#include "CpkMat.h"
#ifndef M_PI
#define M_PI  3.14159265358979323846
#endif

namespace pk
{
	template <class T>
	class TwoValue
	{
	public:
		T x1;
		T x2;
	};
	int normalization(CpkMat&dest,CpkMat&src,double min=0,double max=1);
	double dot(CpkMat& x1,CpkMat& x2);
	CpkMat matDotMat(CpkMat& x1,CpkMat& x2);
	int svd(CpkMat& src,CpkMat& U,CpkMat&V,CpkMat&S,double eps=0.000001);

	//�����л�����ƽ��ֵ,������Ӧ���л��о���
	int avg(CpkMat&dest,CpkMat&src,bool bColumn=true);

	//ĳ�������������ֵ
	CpkMat sub(double num,CpkMat&src);

	int pca(CpkMat&dest,CpkMat&src,bool bColumn=true);

	//������ת180��
	int rot180(CpkMat&dest,CpkMat&src);

	//�������,������matlab������,matlab��vaild�����Ҫ��ת180,���ﶼ����
	int conv2(CpkMat& dest,CpkMat&src,CpkMat&mask,bool bFull);

	CpkMat exp(CpkMat&src);
	CpkMat log(CpkMat&src);
	CpkMat pow(CpkMat&src,int num);

	//�������ȷֲ����������
	CpkMat rand(int row,int col); 
	//������׼��˹�ֲ��ľ���
	CpkMat randn(int row,int col);
	//����1��������ҵ�����,��0��ʼ��num-1Ϊ�������ֵ
	CpkMat randperm(int num);

	//�ı����Ŀ�ߣ����ݲ���
	CpkMat reshape(CpkMat&src,int row,int col);

	//����ֻ����ƽ��
	CpkMat repmat(CpkMat&src,int col);

	//����TwoValue,x1Ϊֵ��x2Ϊ�±�
	TwoValue<double> Max(CpkMat&src);
	//1����ÿ�е����ֵ��2������
	TwoValue<CpkMat> Max(CpkMat&src,int type);

	CpkMat sum(CpkMat&src, int type=1);

	CpkMat kron(CpkMat&src1,CpkMat&src2);

	int save(const char*path,CpkMat&src);
	int load(const char*path,CpkMat&dest);
	

};
#endif