#pragma once
#include"..\Common\CpkMat.h"
namespace pk
{
	class CPCA
	{
	public:
		//�����洢����
		enum DATA_TYPE{DATA_ROW,DATA_COL};
		//PCA��������
		enum TYPE{PCA,PCA_WHILE,ZCA_WHILE};
		CPCA(void);
		virtual ~CPCA(void);
		int Run(CpkMat&src,float rate,TYPE type);
		int Run(CpkMat&src,DATA_TYPE type);
		int Project(CpkMat&dest,CpkMat&src,int eigenNum,DATA_TYPE dataType,TYPE type);
		
		int saveData(const char* path);
		int loadData(const char* path);
	private:
		CpkMat m_projectMat,m_avgMat;
		int avg(CpkMat&dest,CpkMat&src);
		bool m_bInit;
		
	};
};
