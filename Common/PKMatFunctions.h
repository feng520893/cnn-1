#ifndef PK_MAT_FUNCTIONS_H 
#define PK_MAT_FUNCTIONS_H 
#include "CpkMat.h"
#ifndef M_PI
#define M_PI  3.14159265358979323846
#endif

namespace pk
{

	enum DATA_TYPE{DATA_COLS,DATA_ROWS};

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

	//求平均值
	//DATA_ROWS计算每行的平均值返回列向量
	//DATA_COLS计算每列的平均值返回行向量
	int avg(CpkMat&dest,CpkMat&src,DATA_TYPE dataType=DATA_COLS);

	//根据传入的向量生成对角矩阵
	CpkMat diag(CpkMat&src);

	//某个数减矩阵各个值
	CpkMat sub(double num,CpkMat&src);

	//矩阵减向量
	//DATA_ROWS为减行向量
	//DATA_COLS为减列向量
	CpkMat subVec(CpkMat&src,CpkMat&vec,DATA_TYPE dataType=DATA_COLS);

	//矩阵旋转180度
	int rot180(CpkMat&dest,CpkMat&src);

	//卷积操作,这里与matlab有区别,matlab的vaild卷积核要翻转180,这里都不用
	int conv2(CpkMat& dest,CpkMat&src,CpkMat&mask,bool bFull);

	CpkMat exp(CpkMat&src);
	CpkMat log(CpkMat&src);
	CpkMat pow(CpkMat&src,int num);

	//产生均匀分布的随机变量
	CpkMat rand(int row,int col); 
	//产生标准高斯分布的矩阵
	CpkMat randn(int row,int col,float zoomSize=1);
	//生成1个随机打乱的序列,从0开始，num-1为序列最大值
	CpkMat randperm(int num);

	//改变矩阵的宽高，数据不变
	CpkMat reshape(CpkMat&src,int row,int col);

	//返回TwoValue,x1为值，x2为下标
	TwoValue<double> Max(CpkMat&src);
	//1返回每列的最大值，2返回行
	TwoValue<CpkMat> Max(CpkMat&src,int type);

	CpkMat sum(CpkMat&src, int type=1);

	CpkMat kron(CpkMat&src1,CpkMat&src2);

	int save(const char*path,CpkMat&src);
	int load(const char*path,CpkMat&dest);
	

};
#endif