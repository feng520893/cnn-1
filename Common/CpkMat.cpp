#include "CpkMat.h"
#include "mem.h"

CpkMat::CpkMat()
{
	lineSize=Col=Row=Depth=0;
	DataUnion.m_pByte=NULL;//共享内存，初始化化1个就行
	m_refCount=NULL;
	m_dataType=DATA_NULL;
}

CpkMat::CpkMat(const CpkMat& s):Col(s.Col),Row(s.Row),Depth(s.Depth)
,lineSize(s.lineSize),m_refCount(s.m_refCount),m_dataType(s.m_dataType)
{
	if(s.DataUnion.m_pByte!=NULL)
		DataUnion.m_pByte=s.DataUnion.m_pByte;//共享内存，初始化化1个就行
	else
		DataUnion.m_pByte=NULL;
	addRef(1);
}

CpkMat::CpkMat(int row,int col,int depth,DATA_TYPE type,void* data/* =NULL */)
{
	m_refCount=NULL;
	DataUnion.m_pByte=NULL;//共享内存，初始化化1个就行
	Resize(row,col,depth,type,data);
}

CpkMat::~CpkMat()
{
	if(m_refCount&&addRef(-1)==1)
		release();
}

int CpkMat::addRef(int addCount)
{
	if(m_refCount==NULL)
		return 0;
	int tmp=*m_refCount;
	*m_refCount+=addCount;
	return tmp;
}

int CpkMat::release()
{
	if(DataUnion.m_pByte!=NULL)
	{
		switch(m_dataType)
		{
		case DATA_INT:
			fastFree(DataUnion.m_pInt);
			break;
		case DATA_BYTE:
			fastFree(DataUnion.m_pByte);			
			break;
		case DATA_DOUBLE:
			fastFree(DataUnion.m_pDouble);
			break;
		}
		DataUnion.m_pByte=NULL;
	}
	return PK_SUCCESS;
}

CpkMat& CpkMat::operator= (const CpkMat &s)
{
	if(this!=&s)
	{
		if(m_refCount&&addRef(-1)==1)
			release();
		m_refCount=s.m_refCount;
		Row=s.Row;
		Col=s.Col;
		Depth=s.Depth;
		lineSize=s.lineSize;
		m_dataType=s.m_dataType;
		DataUnion.m_pByte=s.DataUnion.m_pByte;//共享内存，初始化化1个就行
		addRef(1);
	}
	return *this;
}

CpkMat CpkMat::operator* (CpkMat& mx1)   
{
//	assert(m_col==mx1.m_row);
	CpkMat mxTmp(Row,mx1.Col,Depth,GetType());   

	switch(m_dataType)
	{
	case DATA_BYTE:
		{
			BYTE* pX1=GetData<BYTE>();
			BYTE* pX2=mx1.GetData<BYTE>();
			BYTE* pDest=mxTmp.GetData<BYTE>();
			for (int i = 0; i <Row; i++)   
			{   
				for(int k=0;k<mx1.Col;k++)
				{
					for (int j = 0; j < Col; j++)   
					{   
						pDest[i*mxTmp.Col+k]+= pX2[j*mx1.Col+k] * pX1[i*Col+j];   
					}   
				}
			}   
		}
		break;
	case DATA_DOUBLE:
		{
			double* pX1=GetData<double>();
			double* pX2=mx1.GetData<double>();
			double* pDest=mxTmp.GetData<double>();
			for (int i = 0; i <Row; i++)   
			{   
				for(int k=0;k<mx1.Col;k++)
				{
					for (int j = 0; j < Col; j++)   
					{   
						pDest[i*mxTmp.Col+k]+= pX2[j*mx1.Col+k] * pX1[i*Col+j];   
					}   
				}
			}
		}
		break;
	case DATA_INT:
		{
			int* pX1=GetData<int>();
			int* pX2=mx1.GetData<int>();
			int* pDest=mxTmp.GetData<int>();
			for (int i = 0; i <Row; i++)   
			{   
				for(int k=0;k<mx1.Col;k++)
				{
					for (int j = 0; j < Col; j++)   
					{   
						pDest[i*mxTmp.Col+k]+= pX2[j*mx1.Col+k] * pX1[i*Col+j];   
					}   
				}
			}   
		}
		break;
	}
	return mxTmp;   
}

CpkMat CpkMat::operator *(double num)
{

	CpkMat temp(Row,Col,Depth,m_dataType);
	switch(m_dataType)
	{
		case DATA_BYTE:
			{
				BYTE* pBuff=temp.GetData<BYTE>();
				for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pByte[i*Col+j]*num;
			}
			break;
		case DATA_DOUBLE:
			{
				double* pBuff=temp.GetData<double>();
				for(int i=0;i<Row;i++)
					for(int j=0;j<Col;j++)
						pBuff[i*Col+j]=DataUnion.m_pDouble[i*Col+j]*num;
			}
			break;
		case DATA_INT:
			{
				int* pBuff=temp.GetData<int>();
				for(int i=0;i<Row;i++)
					for(int j=0;j<Col;j++)
						pBuff[i*Col+j]=DataUnion.m_pInt[i*Col+j]*num;
			}
			break;
	}
	return temp;
}

CpkMat CpkMat::operator /(CpkMat&mx)
{
	CpkMat temp(Row,Col,Depth,m_dataType);
	switch(m_dataType)
	{
	case DATA_BYTE:
		{
			BYTE* pBuff=temp.GetData<BYTE>();
			BYTE* pX2=mx.GetData<BYTE>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pByte[i*Col+j]/pX2[i*mx.Col+j];
		}
		break;
	case DATA_DOUBLE:
		{
			double* pBuff=temp.GetData<double>();
			double* pX2=mx.GetData<double>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pDouble[i*Col+j]/pX2[i*mx.Col+j];
		}
		break;
	case DATA_INT:
		{
			int* pBuff=temp.GetData<int>();
			int* pX2=mx.GetData<int>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pInt[i*Col+j]/pX2[i*mx.Col+j];;
		}
		break;
	}
	return temp;
}

CpkMat CpkMat::operator /(double num)
{
	CpkMat temp(Row,Col,Depth,m_dataType);
	switch(m_dataType)
	{
	case DATA_BYTE:
		{
			BYTE* pBuff=temp.GetData<BYTE>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pByte[i*Col+j]/num;
		}
		break;
	case DATA_DOUBLE:
		{
			double* pBuff=temp.GetData<double>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pDouble[i*Col+j]/num;
		}
		break;
	case DATA_INT:
		{
			int* pBuff=temp.GetData<int>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pInt[i*Col+j]/num;
		}
		break;
	}
	return temp;
}

CpkMat CpkMat::operator /(int num)
{
	CpkMat temp(Row,Col,Depth,m_dataType);
	switch(m_dataType)
	{
	case DATA_BYTE:
		{
			BYTE* pBuff=temp.GetData<BYTE>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pByte[i*Col+j]/num;
		}
		break;
	case DATA_DOUBLE:
		{
			double* pBuff=temp.GetData<double>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pDouble[i*Col+j]/num;
		}
		break;
	case DATA_INT:
		{
			int* pBuff=temp.GetData<int>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pInt[i*Col+j]/num;
		}
		break;
	}
	return temp;
}

CpkMat& CpkMat::operator/=(int num)
{
	switch(m_dataType)
	{
	case DATA_BYTE:
		{
			BYTE* pBuff=this->GetData<BYTE>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]/=num;
		}
		break;
	case DATA_DOUBLE:
		{
			double* pBuff=this->GetData<double>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]/=num;
		}
		break;
	case DATA_INT:
		{
			int* pBuff=this->GetData<int>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]/=num;
		}
		break;
	}
	return *this;
}

CpkMat& CpkMat::operator /=(CpkMat&mx)
{
	switch(m_dataType)
	{
	case DATA_BYTE:
		{
			BYTE* pBuff=this->GetData<BYTE>();
			BYTE* pX2=mx.GetData<BYTE>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]/=pX2[i*mx.Col+j];
		}
		break;
	case DATA_DOUBLE:
		{
			double* pBuff=this->GetData<double>();
			double* pX2=mx.GetData<double>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]/=pX2[i*mx.Col+j];
		}
		break;
	case DATA_INT:
		{
			int* pBuff=this->GetData<int>();
			int* pX2=mx.GetData<int>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]/=pX2[i*mx.Col+j];
		}
		break;
	}
	return *this;
}

CpkMat CpkMat::operator-(CpkMat&mx)
{
	CpkMat temp(Row,Col,Depth,m_dataType);
	if(mx.Row==1)
	{
		double* pBuff=temp.GetData<double>();
		double* pX2=mx.GetData<double>();
		for(int i=0;i<Row;i++)
			for(int j=0;j<Col;j++)
				pBuff[i*Col+j]=DataUnion.m_pDouble[i*Col+j]-pX2[j];
		return temp;
	}

	switch(m_dataType)
	{
	case DATA_BYTE:
		{
			BYTE* pBuff=temp.GetData<BYTE>();
			BYTE* pX2=mx.GetData<BYTE>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pByte[i*Col+j]-pX2[i*mx.Col+j];
		}
		break;
	case DATA_DOUBLE:
		{
			double* pBuff=temp.GetData<double>();
			double* pX2=mx.GetData<double>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pDouble[i*Col+j]-pX2[i*mx.Col+j];
		}
		break;
	case DATA_INT:
		{
			int* pBuff=temp.GetData<int>();
			int* pX2=mx.GetData<int>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pInt[i*Col+j]-pX2[i*mx.Col+j];;
		}
		break;
	}
	return temp;
}

CpkMat CpkMat::operator-(double num)
{
	CpkMat temp(Row,Col,Depth,m_dataType);
	switch(m_dataType)
	{
	case DATA_BYTE:
		{
			BYTE* pBuff=temp.GetData<BYTE>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pByte[i*Col+j]-num;
		}
		break;
	case DATA_DOUBLE:
		{
			double* pBuff=temp.GetData<double>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pDouble[i*Col+j]-num;
		}
		break;
	case DATA_INT:
		{
			int* pBuff=temp.GetData<int>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pInt[i*Col+j]-num;
		}
		break;
	}
	return temp;
}

CpkMat& CpkMat::operator-=(CpkMat&mx)
{
	switch(m_dataType)
	{
	case DATA_BYTE:
		{
			BYTE* pBuff=this->GetData<BYTE>();
			BYTE* pX2=mx.GetData<BYTE>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<lineSize;j++)
					pBuff[i*Col+j]-=pX2[i*mx.lineSize+j];
		}
		break;
	case DATA_DOUBLE:
		{
			double* pBuff=this->GetData<double>();
			double* pX2=mx.GetData<double>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]-=pX2[i*mx.Col+j];
		}
		break;
	case DATA_INT:
		{
			int* pBuff=this->GetData<int>();
			int* pX2=mx.GetData<int>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]-=pX2[i*mx.Col+j];;
		}
		break;
	}
	return *this;
}

CpkMat& CpkMat::operator-=(double num)
{
	switch(m_dataType)
	{
	case DATA_BYTE:
		{
			BYTE* pBuff=this->GetData<BYTE>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<lineSize;j++)
					pBuff[i*lineSize+j]-=num;
		}
		break;
	case DATA_DOUBLE:
		{
			double* pBuff=this->GetData<double>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]-=num;
		}
		break;
	case DATA_INT:
		{
			int* pBuff=this->GetData<int>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]-=num;
		}
		break;
	}
	return *this;
}


CpkMat CpkMat::operator+(CpkMat&mx)
{
	CpkMat temp(Row,Col,Depth,m_dataType);
	switch(m_dataType)
	{
	case DATA_BYTE:
		{
			BYTE* pBuff=temp.GetData<BYTE>();
			BYTE* pX2=mx.GetData<BYTE>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pByte[i*Col+j]+pX2[i*mx.Col+j];
		}
		break;
	case DATA_DOUBLE:
		{
			double* pBuff=temp.GetData<double>();
			double* pX2=mx.GetData<double>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pDouble[i*Col+j]+pX2[i*mx.Col+j];
		}
		break;
	case DATA_INT:
		{
			int* pBuff=temp.GetData<int>();
			int* pX2=mx.GetData<int>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pInt[i*Col+j]+pX2[i*mx.Col+j];;
		}
		break;
	}
	return temp;
}

CpkMat CpkMat::operator+(double num)
{
	CpkMat temp(Row,Col,Depth,m_dataType);
	switch(m_dataType)
	{
	case DATA_BYTE:
		{
			BYTE* pBuff=temp.GetData<BYTE>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pByte[i*Col+j]+num;
		}
		break;
	case DATA_DOUBLE:
		{
			double* pBuff=temp.GetData<double>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pDouble[i*Col+j]+num;
		}
		break;
	case DATA_INT:
		{
			int* pBuff=temp.GetData<int>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=DataUnion.m_pInt[i*Col+j]+num;
		}
		break;
	}
	return temp;
}

CpkMat& CpkMat::operator+=(double num)
{
	switch(m_dataType)
	{
	case DATA_BYTE:
		{
			BYTE* pBuff=this->GetData<BYTE>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<lineSize;j++)
					pBuff[i*Col+j]+=num;
		}
		break;
	case DATA_DOUBLE:
		{
			double* pBuff=this->GetData<double>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]+=num;
		}
		break;
	case DATA_INT:
		{
			int* pBuff=this->GetData<int>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]+=num;
		}
		break;
	}
	return *this;
}

CpkMat& CpkMat::operator+=(CpkMat&mx)
{
	switch(m_dataType)
	{
	case DATA_BYTE:
		{
			BYTE* pBuff=this->GetData<BYTE>();
			BYTE* pX2=mx.GetData<BYTE>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<lineSize;j++)
					pBuff[i*lineSize+j]+=pX2[i*mx.lineSize+j];
		}
		break;
	case DATA_DOUBLE:
		{
			double* pBuff=this->GetData<double>();
			double* pX2=mx.GetData<double>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]+=pX2[i*mx.Col+j];
		}
		break;
	case DATA_INT:
		{
			int* pBuff=this->GetData<int>();
			int* pX2=mx.GetData<int>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]+=pX2[i*mx.Col+j];;
		}
		break;
	}
	return *this;
}

bool CpkMat::operator== (const CpkMat& mx) const
{
	if(this->DataUnion.m_pByte==mx.DataUnion.m_pByte)
		return true;
	return false;
}

void CpkMat::denominator()
{
	if(m_dataType==DATA_DOUBLE)
	{
		for (int i=0;i<Row;i++)
		{
			for(int j=0;j<Col;j++)
				DataUnion.m_pDouble[i*Col+j]=1/DataUnion.m_pDouble[i*Col+j];
		}
	}
}

void CpkMat::AddMinus()
{
	switch(m_dataType)
	{
	case DATA_BYTE:
		{
			BYTE* pBuff=this->GetData<BYTE>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=-pBuff[i*Col+j];
		}
		break;
	case DATA_DOUBLE:
		{
			double* pBuff=this->GetData<double>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=-pBuff[i*Col+j];
		}
		break;
	case DATA_INT:
		{
			int* pBuff=this->GetData<int>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					pBuff[i*Col+j]=-pBuff[i*Col+j];
		}
		break;
	}
}


void CpkMat::print()
{
	for(int i=0;i<Row;i++)
	{
		for(int j=0;j<Col;j++)
		{
			for(int k=0;k<Depth;k++)
			{
				if(m_dataType==DATA_BYTE)
					std::cout<< *(DataUnion.m_pByte+i*Col*Depth+j*Depth+k) << "\t";
				else if(m_dataType==DATA_DOUBLE)
					std::cout<< *(DataUnion.m_pDouble+i*Col*Depth+j*Depth+k) << "\t";
				else
					std::cout<< *(DataUnion.m_pInt+i*Col*Depth+j*Depth+k) << "\t";
			}
		}
		std::cout<< std::endl;
	}
}

int CpkMat::Resize(int row,int col,int depth,DATA_TYPE type,void* data/* =NULL */)
{
	if(data==NULL&&type==DATA_SAME)
	{
		if(this->Row*this->Col*this->Depth!=row*col*depth)
				return PK_NOT_ALLOW_OPERATOR;
		Row=row;
		Col=col;
		Depth=depth;
		if(m_dataType==DATA_BYTE)
			lineSize=lineSize=(col*depth+3)/4*4;
		else
			lineSize=col;
		return PK_SUCCESS;
	}

	if(m_refCount&&addRef(-1)==1)
		release();
	Row=row;
	lineSize=Col=col;
	Depth=depth;
	m_dataType=type;
	
	size_t totalsize=0;
	BYTE* pTmp;
	if(data!=NULL)
	{
		switch(type)
		{
		case DATA_INT:
//			DataUnion.m_pInt=new int[Row*Col*depth];
			totalsize = alignSize(sizeof(int)*Row*Col*depth, (int)sizeof(*m_refCount));
			pTmp= (BYTE*)fastMalloc(totalsize + (int)sizeof(*m_refCount));
			m_refCount = (int*)(pTmp + totalsize);
			*m_refCount = 1;

			DataUnion.m_pInt=(int*)pTmp;
			memcpy(DataUnion.m_pInt,data,sizeof(int)*Row*Col*depth);
			break;
		case DATA_BYTE:
			lineSize=(col*depth+3)/4*4;
//			DataUnion.m_pByte=new BYTE[Row*lineSize];
			totalsize = alignSize(sizeof(BYTE)*Row*lineSize, (int)sizeof(*m_refCount));
			DataUnion.m_pByte= (BYTE*)fastMalloc(totalsize + (int)sizeof(*m_refCount));
			m_refCount = (int*)(DataUnion.m_pByte + totalsize);
			*m_refCount = 1;


			memcpy(DataUnion.m_pByte,data,sizeof(BYTE)*Row*lineSize);
			break;
		case DATA_DOUBLE:
//			DataUnion.m_pDouble=new double[Row*Col*depth];
			totalsize = alignSize(sizeof(double)*Row*Col*depth, (int)sizeof(*m_refCount));
			pTmp= (BYTE*)fastMalloc(totalsize + (int)sizeof(*m_refCount));
			m_refCount = (int*)(pTmp + totalsize);
			*m_refCount = 1;
			DataUnion.m_pDouble=(double*)pTmp;

			memcpy(DataUnion.m_pDouble,data,sizeof(double)*Row*Col*depth);
			break;
		}
	}
	else
	{
		switch(type)
		{
		case DATA_INT:
//			DataUnion.m_pInt=new int[Row*Col*depth];
			totalsize = alignSize(sizeof(int)*Row*Col*depth, (int)sizeof(*m_refCount));
			pTmp= (BYTE*)fastMalloc(totalsize + (int)sizeof(*m_refCount));
			m_refCount = (int*)(pTmp + totalsize);
			*m_refCount = 1;
			DataUnion.m_pInt=(int*)pTmp;

			memset(DataUnion.m_pInt,0,sizeof(int)*Row*Col*depth);
			break;
		case DATA_BYTE:
			lineSize=(col*depth+3)/4*4;
//			DataUnion.m_pByte=new BYTE[Row*lineSize];
			totalsize = alignSize(sizeof(BYTE)*Row*lineSize, (int)sizeof(*m_refCount));
			DataUnion.m_pByte= (BYTE*)fastMalloc(totalsize + (int)sizeof(*m_refCount));
			m_refCount = (int*)(DataUnion.m_pByte + totalsize);
			*m_refCount = 1;
			memset(DataUnion.m_pByte,0,sizeof(BYTE)*Row*lineSize);
			break;
		case DATA_DOUBLE:
//			DataUnion.m_pDouble=new double[Row*Col*depth];
			totalsize = alignSize(sizeof(double)*Row*Col*depth, (int)sizeof(*m_refCount));
			pTmp= (BYTE*)fastMalloc(totalsize + (int)sizeof(*m_refCount));
			m_refCount = (int*)(pTmp + totalsize);
			*m_refCount = 1;
			DataUnion.m_pDouble=(double*)pTmp;
			memset(DataUnion.m_pDouble,0,sizeof(double)*Row*Col*depth);
			break;
		}
	}
	return PK_SUCCESS;
}

int CpkMat::Resize(DATA_TYPE type)
{
	if(m_dataType==type)
		return PK_NOT_ALLOW_OPERATOR;
	switch(type)
	{
	case DATA_DOUBLE:
		{
			double* pData=new double[Row*lineSize];
			for(int i=0;i<Row*Col;i++)
				pData[i]=(double)DataUnion.m_pByte[i];
			Resize(Row,Col,Depth,type,pData);
			delete [] pData;
			break;
		}
	case DATA_BYTE:
		{
			BYTE* pData=new BYTE[Row*lineSize];
			for(int i=0;i<Row*Col;i++)
				pData[i]=(BYTE)abs(DataUnion.m_pDouble[i]);
			Resize(Row,Col,Depth,type,pData);
			delete [] pData;
		}
	}

	return PK_SUCCESS;
}

int CpkMat::copyTo(CpkMat& newMat,DATA_TYPE newType)
{
	if(newMat==*this)
	{
		newMat=*this;
		return PK_SUCCESS;
	}

	if(m_dataType==newType)
		newMat.Resize(Row,Col,Depth,newType,DataUnion.m_pByte);
	else
	{
		newMat.Resize(Row,Col,Depth,newType);
		switch(m_dataType)
		{
		case DATA_INT:
			{
				switch(newType)
				{
				case DATA_BYTE:
					{
						BYTE* pNew=newMat.GetData<BYTE>();
						for(int i=0;i<Row*Col*Depth;i++)
							*(pNew++)=*(DataUnion.m_pInt++);
						break;
					}
				case DATA_DOUBLE:
					double* pNew=newMat.GetData<double>();
					for(int i=0;i<Row*Col*Depth;i++)
						*(pNew++)=*(DataUnion.m_pInt++);
					break;
				}
				break;
		case DATA_BYTE:
			{
				switch(newType)
				{
				case DATA_INT:
					{
						int* pNew=newMat.GetData<int>();
						for(int i=0;i<Row*Col*Depth;i++)
							*(pNew++)=*(DataUnion.m_pByte++);
						break;
					}
				case DATA_DOUBLE:
					double* pNew=newMat.GetData<double>();
					for(int i=0;i<Row*Col*Depth;i++)
						*(pNew++)=*(DataUnion.m_pByte++);
					break;
				}
				break;
			}
		case DATA_DOUBLE:
			{
				switch(newType)
				{
				case DATA_INT:
					{
						int* pNew=newMat.GetData<int>();
						for(int i=0;i<Row*Col*Depth;i++)
							*(pNew++)=*(DataUnion.m_pDouble++);
						break;
					}
				case DATA_BYTE:
					BYTE* pNew=newMat.GetData<BYTE>();
					for(int i=0;i<Row*Col*Depth;i++)
						*(pNew++)=*(DataUnion.m_pDouble++);
					break;
				}
				break;
			}
			}
		}
	}
	return PK_SUCCESS;
}

int CpkMat::copyTo(CpkMat& newMat,int row,int col,DATA_TYPE newType)
{
	if(row>Row||row<0||col>Col||col<0)
		return PK_NOT_ALLOW_OPERATOR;
	newMat.Resize(row,col,Depth,m_dataType);
	switch(m_dataType)
	{
	case DATA_DOUBLE:
		{
			switch(newType)
			{
			case DATA_DOUBLE:
				double* pNew=newMat.GetData<double>();
				for(int i=0;i<row;i++)
					for(int j=0;j<col;j++)
						pNew[i*col+j]=DataUnion.m_pDouble[i*Col+j];
				break;
			}
			break;
		}
	}
	return PK_SUCCESS;
}

CpkMat CpkMat::ColumnVector()
{
	CpkMat tmp;
	ColumnVector(tmp);
	return tmp;
}

int CpkMat::ColumnVector(CpkMat& dest)
{
	if(DataUnion.m_pByte==NULL)
		return PK_NOT_ALLOW_OPERATOR;
	dest.Resize(Row*Col,1,Depth,GetType());
	switch(m_dataType)
	{
	case DATA_BYTE:
		{
			BYTE* tempBuff=DataUnion.m_pByte;
		BYTE* pData=dest.GetData<BYTE>();
			for(int i=0;i<Col;i++)
				for(int j=0;j<Row;j++)
				*(pData++)=tempBuff[j*Col+i];
		}
		break;
	case DATA_DOUBLE:
		{
			double* tempBuff=DataUnion.m_pDouble;
			double* pData=dest.GetData<double>();
			for(int i=0;i<Col;i++)
				for(int j=0;j<Row;j++)
					*(pData++)=tempBuff[j*Col+i];
		}
		break;
	case DATA_INT:
		{
			int* tempBuff=DataUnion.m_pInt;
			int* pData=dest.GetData<int>();
			for(int i=0;i<Col;i++)
				for(int j=0;j<Row;j++)
				*(pData++)=tempBuff[j*Col+i];
		}
		break;
	}
	return PK_SUCCESS;
}

CpkMat& CpkMat::RowVector()
{
	Col=Row*Col;
	lineSize=Col*Depth;
	Row=1;
	return *this;
}
int CpkMat::RowVector(CpkMat& dest)
{
	switch(m_dataType)
	{
	case DATA_BYTE:
		dest.Resize(1,Row*Col,Depth,m_dataType,DataUnion.m_pByte);
		break;
	case DATA_DOUBLE:
		if(dest.Row!=0||dest.Col!=0)
		{
			double* pDest=dest.GetData<double>();
			for(int i=0;i<Row;i++)
				for(int j=0;j<Col;j++)
					*(pDest++)=DataUnion.m_pDouble[i*Col+j];
		}
		else
			dest.Resize(1,Row*Col,Depth,m_dataType,DataUnion.m_pDouble);
		break;
	case DATA_INT:
		dest.Resize(1,Row*Col,Depth,m_dataType,DataUnion.m_pInt);
		break;
	}
	return PK_SUCCESS;
}

int CpkMat::setAllData(int data)
{
	if(DataUnion.m_pByte==NULL)
		return PK_NOT_ALLOW_OPERATOR;
	switch(m_dataType)
	{
	case DATA_INT:
		for (int i = 0; i <Row; i++)   
		{   
			for (int j = 0; j <Col; j++)   
			{   
				DataUnion.m_pInt[i*Col+j]=data;   
			}   
		}   
		break;
	case DATA_BYTE:
		for (int i = 0; i <Row; i++)   
		{   
			for (int j = 0; j <Col; j++)   
			{   
				DataUnion.m_pByte[i*Col+j]=data;   
			}   
		}   
		break;
	case DATA_DOUBLE:
			for (int i = 0; i <Row; i++)   
		{   
			for (int j = 0; j <Col; j++)   
			{   
				DataUnion.m_pDouble[i*Col+j]=data;   
			}   
		}   
		break;
	}
	return PK_SUCCESS;
}

int CpkMat::setData(CpkMat&src,int rowS,int rowE,int colS,int colE)
{
	int dCol=colE-colS;
	int dRow=rowE-rowS;

	if(dCol>lineSize||dRow>Row)
		return PK_NOT_ALLOW_OPERATOR;

	switch(m_dataType)
	{
	case DATA_DOUBLE:
		{
			double * pSrc=src.GetData<double>();
			for(int i=0;i<dRow;i++)
			{
				for(int j=0;j<dCol;j++)
				{
					for(int n=0;n<Depth;n++)
						DataUnion.m_pDouble[(i+rowS)*lineSize+j+colS+n]=pSrc[i*src.lineSize+j+n];
				}
			}
			break;
		}
	case DATA_BYTE:
		{
			BYTE * pSrc=src.GetData<BYTE>();
			for(int i=0;i<dRow;i++)
			{
				for(int j=0;j<src.lineSize;j++)
					DataUnion.m_pByte[(i+rowS)*lineSize+j+colS*Depth]=pSrc[i*src.lineSize+j];
			}
			break;
		}
	case DATA_INT:
		{
			int * pSrc=src.GetData<int>();
			for(int i=0;i<=rowE-rowS;i++)
			{
				for(int j=0;j<=colE-colS;j++)
				{
					for(int n=0;n<Depth;n++)
						DataUnion.m_pInt[(i+rowS)*lineSize+j+colS+n]=pSrc[i*src.lineSize+j+n];
				}
			}
			break;
		}
	}
	return PK_SUCCESS;
}

int CpkMat::GetData(CpkMat& dest,int rowS,int rowE,int colS,int colE)
{
	int dCol=colE-colS;
	int dRow=rowE-rowS;

	if(colE>lineSize||dRow>Row)
		return PK_NOT_ALLOW_OPERATOR;

	switch(m_dataType)
	{
	case DATA_INT:
		{
			dest.Resize(rowE-rowS,dCol,Depth,m_dataType);
			int * pdata=dest.GetData<int>();
			int * pSrc=DataUnion.m_pInt;
			for(int i=0;i<rowE-rowS;i++)
				memcpy(pdata+i*dest.Col*Depth,pSrc+(i+rowS)*Col*Depth+colS,sizeof(int)*dCol*Depth);
			break;
		}
	case DATA_BYTE:
		{
			int lineOut=0;
			if(colS==Col-1)
				lineOut=lineSize-(colS*Depth+3)/4*4;
			else
				lineOut=dCol*Depth;
			dest.Resize(dRow,dCol,Depth,m_dataType);
			dest.lineSize=lineOut;

			BYTE * pdata=dest.GetData<BYTE>();
			BYTE * pSrc=DataUnion.m_pByte;
			
			for(int i=0;i<dRow;i++)
				for(int j=0;j<lineOut;j++)
					pdata[i*lineOut+j]=pSrc[(i+rowS)*lineSize+j+colS*Depth];
			break;
		}
	case DATA_DOUBLE:
		{
			dest.Resize(rowE-rowS,dCol,Depth,m_dataType);
			double * pSrc=DataUnion.m_pDouble;
			double* pdata=dest.GetData<double>();
			for(int i=0;i<rowE-rowS;i++)
			{
//				for(int j=0;j<(colE-colS)*Depth;j++)
//				{
//					for(int n=0;n<Depth;n++)
//						pdata[i*dCol+j*Depth+n]=pSrc[(i+rowS)*Col+(j+colS)*Depth+n];
//					pdata[i*dest.Col*Depth+j]=pSrc[(i+rowS)*Col*Depth+j+colS];
//				}
				memcpy(pdata+i*dest.Col*Depth,pSrc+(i+rowS)*Col*Depth+colS,sizeof(double)*dCol*Depth);
			}
			break;
		}
	}
	return PK_SUCCESS;
}

CpkMat CpkMat::Transpose()
{
	CpkMat tmp;
	int nRet=Transpose(tmp);
	return tmp;
}

int CpkMat::Transpose(CpkMat& mxTmp)
{
	mxTmp.Resize(Col,Row,Depth,m_dataType);
	switch(m_dataType)
	{
	case DATA_BYTE:
		{
			BYTE* pBuff=mxTmp.GetData<BYTE>();
			for (int i = 0; i <Row; i++)   
			{   
				for (int j = 0; j <Col; j++)   
				{   
					pBuff[j*mxTmp.Col+i]=DataUnion.m_pByte[i*Col+j];   
				}   
			}   
		}
		break;
	case DATA_DOUBLE:
		{
			double* pBuff=mxTmp.GetData<double>();
			for (int i = 0; i <Row; i++)   
			{   
				for (int j = 0; j <Col; j++)   
				{   
					pBuff[j*mxTmp.Col+i]=DataUnion.m_pDouble[i*Col+j];   
				}   
			}   
		}
		break;
	case DATA_INT:
		{
			int* pBuff=mxTmp.GetData<int>();
			for (int i = 0; i <Row; i++)   
			{   
				for (int j = 0; j <Col; j++)   
				{   
					pBuff[j*mxTmp.Col+i]=DataUnion.m_pInt[i*Col+j];   
				}   
			}   
		}
		break;
	}
	
	return 0;
}


void CpkMat::zeros()
{
	switch(m_dataType)
	{
	case DATA_BYTE:
			memset(DataUnion.m_pByte,0,sizeof(BYTE)*Row*Col*Depth);
		break;
	case DATA_DOUBLE:
		memset(DataUnion.m_pDouble,0,sizeof(double)*Row*Col*Depth);
		break;
	case DATA_INT:
		memset(DataUnion.m_pInt,0,sizeof(int)*Row*Col*Depth);
		break;
	}
}