#pragma once
#include"PKDefine.h"
#include <iostream>
class CpkMat
{
public:
	enum DATA_TYPE{DATA_NULL,DATA_SAME,DATA_BYTE,DATA_DOUBLE,DATA_INT};
	CpkMat(void);
	CpkMat(const CpkMat& s);
	CpkMat(int row,int col,int depth,DATA_TYPE type,void* data=NULL);
	virtual ~CpkMat(void);

	int Resize(int row,int col,int depth,DATA_TYPE type=DATA_SAME,void* data=NULL);
	int Resize(DATA_TYPE type);
	void print();

	//תΪ������
	CpkMat ColumnVector();
	int ColumnVector(CpkMat& dest);

	//תΪ������
	CpkMat& RowVector();
	int RowVector(CpkMat& dest);

	//����ת��
	int Transpose(CpkMat&dest);
	CpkMat Transpose();

	//��ɷ���ʽ��c++�Ҳ���1/�ķ���
	void denominator();

	//��Ӹ���
	void AddMinus();

	//�Ѿ�����Ϊȫ0����
	void zeros();

	bool Empty()
	{
		return m_dataType==DATA_NULL?true:false;
	}

	DATA_TYPE GetType()
	{
		return m_dataType;
	};

	template<typename T>
	T& at(int rowIndex,int colIndex)
	{
		switch(m_dataType)
		{
		case DATA_INT:
			return (T&)*(DataUnion.m_pInt+rowIndex*lineSize+colIndex*Depth);
		case DATA_DOUBLE:
			return (T&)*(DataUnion.m_pDouble+rowIndex*lineSize+colIndex*Depth);
		}
		return (T&)*(DataUnion.m_pByte+rowIndex*lineSize+colIndex*Depth);
	}

	template<typename T>
	T* GetData(int nRow=0)
	{
		switch(m_dataType)
		{
		case DATA_INT:
			return (T*)(DataUnion.m_pInt+nRow*Col*Depth);
		case DATA_BYTE:
			return (T*)(DataUnion.m_pByte+nRow*lineSize);
		case DATA_DOUBLE:
			return (T*)(DataUnion.m_pDouble+nRow*Col*Depth);
		}
		return NULL;
	}
	int GetData(CpkMat& dest,int rowS,int rowE,int colS,int colE);

	int copyTo(CpkMat& newMat,DATA_TYPE newType);
	int copyTo(CpkMat& newMat,int row,int col,DATA_TYPE newType);

	int setAllData(int data);

	int setData(CpkMat&src,int rowS,int rowE,int colS,int colE);

	CpkMat& operator= (const CpkMat &mx);
	CpkMat operator*(CpkMat&mx);
	CpkMat operator*(double num);
	CpkMat operator/(CpkMat&mx);
	CpkMat operator/(double num);
	CpkMat operator/(int num);
	CpkMat& operator/=(CpkMat&mx);
	CpkMat& operator/=(int num);
	CpkMat operator-(CpkMat&mx);
	CpkMat operator-(double mx);
	CpkMat& operator-=(CpkMat&mx);
	CpkMat& operator-=(double mx);
	CpkMat operator+(CpkMat&mx);
	CpkMat operator+(double num);
	CpkMat& operator+=(CpkMat&mx);
	CpkMat& operator+=(double num);
	bool operator== (const CpkMat& mx) const;

public:
	int Col;
	int Row;
	int lineSize;     //����ͼƬ����Ŀ��
	short Depth;
private:
	DATA_TYPE m_dataType;
	union _dataUnion
	{ 
		BYTE* m_pByte;
		double* m_pDouble;
		int*  m_pInt;
	}DataUnion;

	int* m_refCount;

	int addRef(int addCount); 
	int release();
};
