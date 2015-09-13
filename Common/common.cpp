#include"common.h"
#include<stdlib.h>
#include<io.h>
namespace pk
{
	void transpose(double** data,int& row,int& col)
	{
		double* pTmp=*data;
		*data=new double[row*col];
		for (int i = 0; i <row; i++)   
		{   
			for (int j = 0; j <col; j++)   
				(*data)[j*row+i]=pTmp[i*col+j];   
		}
		int tmp=col;
		col=row;
		row=tmp;
		delete [] pTmp;
	}  

	double* matrixMul(double* left,int leftRow,int leftCol,double* right,int rightRow,int rightCol)
	{
		double* pX1=left;
		double* pX2=right;
		double* pDest=new double[leftRow*rightCol];
		memset(pDest,0,sizeof(double)*leftRow*rightCol);
		for (int i = 0; i <leftRow; i++)   
		{   
			for(int k=0;k<rightCol;k++)
			{
				for (int j = 0; j <leftCol; j++)   
				{   
					pDest[i*rightCol+k]+= pX2[j*rightCol+k] * pX1[i*leftCol+j];   
				}   
			}
		}   
		return pDest;
	}

	int findDirectsOrFiles(std::string direct,std::vector<std::string>& files,bool bOnlyFindDirect)
	{
		std::string path=direct+std::string("\\*.*");
		long handle;  

		struct _finddata_t fileinfo;
		handle=_findfirst(path.c_str(),&fileinfo);  
		if(-1==handle)
			return PK_NOT_FILE;   
		while(!_findnext(handle,&fileinfo))  
		{  
			if(strcmp(fileinfo.name,".")==0||strcmp(fileinfo.name,"..")==0)
				continue;                                                         
			if(bOnlyFindDirect&&fileinfo.attrib!=_A_SUBDIR)  
				continue;   
			std::string destFile=direct+std::string("\\")+std::string(fileinfo.name);  
			files.push_back(destFile);
		}  
		_findclose(handle);  
		return PK_SUCCESS;
	}
	
}