#include "PCA.h"
#include"..\Common\PKMatFunctions.h"
#include <fstream>
namespace pk
{

	CPCA::CPCA(void)
	{
		m_bInit=false;
	}


	CPCA::~CPCA(void)
	{
	}

	int CPCA::Run(CpkMat&src,float rate,TYPE type)
	{
		CpkMat temp,u,v,s,cov;
		if(type==DATA_COL)
			src.Transpose(temp);
		else
			temp=src;
		avg(m_avgMat,temp);

		u.Resize(src.Row,1,1,m_avgMat.GetType());
		u.setAllData(1);
		u=u*m_avgMat;
		
		//svd
		CpkMat tempT;
		tempT=s=temp-u;
		s.Transpose(v);
		//当N*M矩阵,M>>N时,用A*A'替代A'*A,对应的特征向量为X=A'*Y。  X为A'*A,Y为A*A'
		if(s.Col>s.Row*100)
		{
			cov=s*v/temp.Row;
			pk::svd(cov,u,v,s);
			tempT.Transpose(u);
			v=u*v;
		}
		else
		{
			cov=v*s/temp.Row;
			pk::svd(cov,u,v,s);
		}

		int min=s.Col<s.Row?s.Col:s.Row;
		double* pData=s.GetData<double>();
		double sum=0;
		double tSum=0;
		for(int i=0;i<min;i++)
		{
			sum+=*(pData+i*s.Col+i);
		}
		int index=0;
		for(int i=0;i<min;i++)
		{
			++index;
			tSum+=pData[i*s.Col+i];
			if(tSum/sum>rate)
				break;
		}

		v.copyTo(m_projectMat,v.Row,index,CpkMat::DATA_DOUBLE);
		return PK_SUCCESS;
	}

	int CPCA::Run(CpkMat&src,DATA_TYPE type)
	{
		CpkMat temp,u,v,s,cov;

		if(type==DATA_ROW)
			temp=src.Transpose();
		else
			temp=src;
		avg(m_avgMat,temp);

		u=subVecRow(temp,m_avgMat);
		//svd
		CpkMat tempT;
		tempT=s=u;
		v=s.Transpose();
		//当N*M矩阵,M>>N时,用A*A'替代A'*A,对应的特征向量为X=A'*Y。  X为A'*A,Y为A*A'
/*		if(s.Col>s.Row*100)
		{
			cov=s*v/temp.Row;
			pk::svd(cov,u,v,s);
			tempT.Transpose(u);
			v=u*v;
		}
		else
		{
			cov=v*s/temp.Row;
			pk::svd(cov,u,v,s);
		}*/
		cov=s*v/src.Col;

		pk::svd(cov,m_projectMat,v,s);
		return PK_SUCCESS;
	}



	int CPCA::Project(CpkMat&dest,CpkMat&src,int eigenNum,DATA_TYPE dataType,TYPE type)
	{
		if(!m_bInit)
			return PK_NOT_INIT;
		if(eigenNum<1||eigenNum>src.Row)
			return PK_NOT_ALLOW_OPERATOR;

		CpkMat temp,tempDest;
		if(dataType==DATA_COL)
			src.Transpose(temp);
		else
			temp=src;
		CpkMat sample;
		tempDest.Resize(src.Row,m_projectMat.Col,1,m_projectMat.GetType());
		for(int i=0;i<temp.Row;i++)
		{
			temp.getRow(sample,i);
			sample=sample-m_avgMat;
			sample=sample*m_projectMat;
			tempDest.setRowData(i,sample);
		}
		if(dataType==DATA_COL)
		{
			tempDest.Transpose(sample);
			dest=sample;
		}
		else
			dest=tempDest;
		return PK_SUCCESS;
	}

	int CPCA::avg(CpkMat&dest,CpkMat&src)
	{
		switch(src.GetType())
		{
		case CpkMat::DATA_DOUBLE:
			{
				dest.Resize(1,src.Col,1,src.GetType());
				double* buffer=dest.GetData<double>();
				double* pData=src.GetData<double>();
				for(int i=0;i<src.Col;i++)
				{	
					for(int j=0;j<src.Row;j++)
						buffer[i]+=pData[j*src.Col+i];
					buffer[i]/=src.Col;
				}
			}
			break;
		case CpkMat::DATA_BYTE:
			{
				dest.Resize(1,src.Col,1,src.GetType());
				BYTE* buffer=dest.GetData<BYTE>();
				BYTE* pData=src.GetData<BYTE>();
				for(int i=0;i<src.Col;i++)
				{	
					for(int j=0;j<src.Row;j++)
						buffer[i]+=pData[j*src.Col+i];
					buffer[i]/=src.Col;
				}
			}
			break;
		case CpkMat::DATA_INT:
			{
				dest.Resize(1,src.Col,1,src.GetType());
				int* buffer=dest.GetData<int>();
				int* pData=src.GetData<int>();
				for(int i=0;i<src.Col;i++)
				{	
					for(int j=0;j<src.Row;j++)
						buffer[i]+=pData[j*src.Col+i];
					buffer[i]/=src.Col;
				}
			}
		}
		return PK_SUCCESS;
	}

	int CPCA::saveData(const char* path)
	{
		FILE* fp;
		fp=fopen(path,"wb");
		if(!fp)
			return PK_OPEN_FILE_ERROR;
		int nRow=m_avgMat.Row;
		fwrite(&nRow,1,sizeof(nRow),fp);
		int nCol=m_avgMat.Col;
		fwrite(&nCol,1,sizeof(nCol),fp);
		short nDepth=m_avgMat.Depth;
		fwrite(&nDepth,1,sizeof(nDepth),fp);
		CpkMat::DATA_TYPE nType=m_avgMat.GetType();
		fwrite(&nType,1,sizeof(nType),fp);
		switch(nType)
		{
		case CpkMat::DATA_DOUBLE:
			{
				double* pBuff=m_avgMat.GetData<double>();
				fwrite(pBuff,1,sizeof(double)*nRow*nCol*nDepth,fp);
			}
			break;
		}

		nRow=m_projectMat.Row;
		fwrite(&nRow,1,sizeof(nRow),fp);
		nCol=m_projectMat.Col;
		fwrite(&nCol,1,sizeof(nCol),fp);
		nDepth=m_projectMat.Depth;
		fwrite(&nDepth,1,sizeof(nDepth),fp);
		nType=m_projectMat.GetType();
		fwrite(&nType,1,sizeof(nType),fp);
		switch(nType)
		{
		case CpkMat::DATA_DOUBLE:
			{
				double* pBuff=m_projectMat.GetData<double>();
				fwrite(pBuff,1,sizeof(double)*nRow*nCol*nDepth,fp);
			}
			break;
		}
		fclose(fp);

		return PK_SUCCESS;
	}

	int CPCA::loadData(const char* path)
	{
		FILE* fp;
		fp=fopen(path,"rb+");
		if(fp==NULL)
			return PK_NOT_FILE;
		int nRow;
		fread(&nRow,1,sizeof(nRow),fp);
		int nCol;
		fread(&nCol,1,sizeof(nCol),fp);
		short nDepth;
		fread(&nDepth,1,sizeof(nDepth),fp);
		CpkMat::DATA_TYPE nType;
		fread(&nType,1,sizeof(nType),fp);
		m_avgMat.Resize(nRow,nCol,nDepth,nType);
		switch(nType)
		{
		case CpkMat::DATA_DOUBLE:
			{
				double* pBuff=m_avgMat.GetData<double>();
				fread(pBuff,1,sizeof(double)*nRow*nCol*nDepth,fp);
			}
			break;
		}
		fread(&nRow,1,sizeof(nRow),fp);
		fread(&nCol,1,sizeof(nCol),fp);
		fread(&nDepth,1,sizeof(nDepth),fp);
		fread(&nType,1,sizeof(nType),fp);
		m_projectMat.Resize(nRow,nCol,nDepth,nType);
		switch(nType)
		{
		case CpkMat::DATA_DOUBLE:
			{
				double* pBuff=m_projectMat.GetData<double>();
				fread(pBuff,1,sizeof(double)*nRow*nCol*nDepth,fp);
			}
			break;
		}
		fclose(fp);
		return PK_SUCCESS;
	}
};