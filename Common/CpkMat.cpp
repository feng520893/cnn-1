#include "CpkMat.h"

CpkMat::CpkMat()
{
	Col=Row=Depth=0;
	DataUnion.m_pByte=NULL;//共享内存，初始化化1个就行
}

CpkMat::CpkMat(const CpkMat& s)
{
	DataUnion.m_pByte=NULL;//共享内存，初始化化1个就行
	Resize(s.Row,s.Col,s.Depth,s.m_dataType,s.DataUnion.m_pByte);
}

CpkMat::CpkMat(int row,int col,int depth,DATA_TYPE type,void* data/* =NULL */)
{
	DataUnion.m_pByte=NULL;//共享内存，初始化化1个就行
	Resize(row,col,depth,type,data);
}

CpkMat::~CpkMat()
{
	if(DataUnion.m_pByte!=NULL)
	{
		switch(m_dataType)
		{
		case DATA_INT:
			delete [] DataUnion.m_pInt;
			break;
		case DATA_BYTE:
			delete [] DataUnion.m_pByte;
			break;
		case DATA_DOUBLE:
			delete [] DataUnion.m_pDouble;
			break;
		}
	}
}

CpkMat& CpkMat::operator= (const CpkMat &s)
{
	Resize(s.Row,s.Col,s.Depth,s.m_dataType,s.DataUnion.m_pByte);
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
	CpkMat temp(Row,Col,1,m_dataType);
	if(mx.Row==1)
	{
		double* pBuff=temp.GetData<double>();
		double* pTmp=mx.GetData<double>();
		for(int i=0;i<Row;i++)
			for(int j=0;j<Col;j++)
				pBuff[i*Col+j]=DataUnion.m_pDouble[i*Col+j]/pTmp[j];
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
	Row=row;
	lineSize=Col=col;
	Depth=depth;
	m_dataType=type;
	if(DataUnion.m_pByte!=NULL)
	{
		switch(type)
		{
		case DATA_INT:
			delete [] DataUnion.m_pInt;
			DataUnion.m_pInt=NULL;
			break;
		case DATA_BYTE:
			delete [] DataUnion.m_pByte;
			DataUnion.m_pByte=NULL;
			break;
		case DATA_DOUBLE:
			delete [] DataUnion.m_pDouble;
			DataUnion.m_pDouble=NULL;
			break;
		}
	}
	if(data!=NULL)
	{
		switch(type)
		{
		case DATA_INT:
			DataUnion.m_pInt=new int[Row*Col*depth];
			memcpy(DataUnion.m_pInt,data,sizeof(int)*Row*Col*depth);
			break;
		case DATA_BYTE:
			lineSize=(col*depth+3)/4*4;
			DataUnion.m_pByte=new BYTE[Row*lineSize];
			memcpy(DataUnion.m_pByte,data,sizeof(BYTE)*Row*lineSize);
			break;
		case DATA_DOUBLE:
			DataUnion.m_pDouble=new double[Row*Col*depth];
			memcpy(DataUnion.m_pDouble,data,sizeof(double)*Row*Col*depth);
			break;
		}
	}
	else
	{
		switch(type)
		{
		case DATA_INT:
			DataUnion.m_pInt=new int[Row*Col*depth];
			memset(DataUnion.m_pInt,0,sizeof(int)*Row*Col*depth);
			break;
		case DATA_BYTE:
			lineSize=(col*depth+3)/4*4;
			DataUnion.m_pByte=new BYTE[Row*lineSize];
			memset(DataUnion.m_pByte,0,sizeof(BYTE)*Row*lineSize);
			break;
		case DATA_DOUBLE:
			DataUnion.m_pDouble=new double[Row*Col*depth];
			memset(DataUnion.m_pDouble,0,sizeof(double)*Row*Col*depth);
			break;
		}
	}
	return PK_SUCCESS;
}

int CpkMat::Resize(DATA_TYPE type)
{
	if(m_dataType==type||m_dataType==DATA_DOUBLE)
		return PK_NOT_ALLOW_OPERATOR;
	double* pData=new double[Row*Col];
	for(int i=0;i<Row*Col;i++)
		pData[i]=(double)DataUnion.m_pByte[i];
	this->Resize(Row,Col,Depth,DATA_DOUBLE,pData);
	delete [] pData;

	return PK_SUCCESS;
}

int CpkMat::copyTo(CpkMat& newMat,DATA_TYPE newType)
{
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

int CpkMat::RowVector()
{
	Col=Row*Col*Depth;
	Row=1;
	return PK_SUCCESS;
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

int CpkMat::setRowData(int nRow,void* data)
{
	if(nRow>=Row)
		return PK_NOT_ALLOW_OPERATOR;
	switch(m_dataType)
	{
	case DATA_INT:
		memcpy(DataUnion.m_pInt+nRow*Col*Depth,data,Col*sizeof(int));
		break;
	case DATA_BYTE:
		memcpy(DataUnion.m_pByte+nRow*Col*Depth,data,Col*sizeof(BYTE));
		break;
	case DATA_DOUBLE:
		memcpy(DataUnion.m_pDouble+nRow*Col*Depth,data,Col*sizeof(double));
	}
	return PK_SUCCESS;
}

int CpkMat::setRowData(int nRow,CpkMat& srcMat)
{
	if(nRow>=Row)
		return PK_NOT_ALLOW_OPERATOR;
	switch(m_dataType)
	{
	case DATA_INT:
		{
			int* pBuff=srcMat.GetData<int>();
			setRowData(nRow,pBuff);
		}
		break;
	case DATA_BYTE:
		{
			BYTE* pBuff=srcMat.GetData<BYTE>();
			setRowData(nRow,pBuff);
		}
		break;
	case DATA_DOUBLE:
		{
			double* pBuff=srcMat.GetData<double>();
			setRowData(nRow,pBuff);
		}
		break;
	}
	return PK_SUCCESS;
}

int CpkMat::setColData(int nCol,void* data)
{
	if(nCol>=Col)
		return PK_NOT_ALLOW_OPERATOR;
	switch(m_dataType)
	{
	case DATA_INT:
		{
			int* pData=(int*)data;
			for(int i=0;i<Row;i++)
				DataUnion.m_pInt[i*Col+nCol]=*(pData++);
		}
		break;
	case DATA_BYTE:
		{
			BYTE* pData=(BYTE*)data;
			for(int i=0;i<Row;i++)
				DataUnion.m_pByte[i*Col+nCol]=*(pData++);
		}
		break;
	case DATA_DOUBLE:
		{
			double* pData=(double*)data;
			for(int i=0;i<Row;i++)
				DataUnion.m_pDouble[i*Col+nCol]=*(pData++);
		}
	}
	return PK_SUCCESS;
}

int CpkMat::setColData(int nCol,CpkMat& srcMat)
{
	if(nCol>=Col||srcMat.Col!=1)
		return PK_NOT_ALLOW_OPERATOR;
	switch(m_dataType)
	{
	case DATA_INT:
		{
			switch(srcMat.GetType())
			{
				case DATA_BYTE:
				{
					BYTE* pData=srcMat.GetData<BYTE>();
					for(int i=0;i<Row;i++)
						DataUnion.m_pInt[i*Col+nCol]=*(pData++);
				}
				break;
				case DATA_DOUBLE:
					{
						double* pData=srcMat.GetData<double>();
						for(int i=0;i<Row;i++)
							DataUnion.m_pInt[i*Col+nCol]=*(pData++);
					}
					break;
				case DATA_INT:
					{
						int* pData=srcMat.GetData<int>();
						for(int i=0;i<Row;i++)
							DataUnion.m_pInt[i*Col+nCol]=*(pData++);
					}
					break;
			}
			break;
			
		}
		break;
	case DATA_BYTE:
		{
			switch(srcMat.GetType())
			{
			case DATA_BYTE:
				{
					BYTE* pData=srcMat.GetData<BYTE>();
					for(int i=0;i<Row;i++)
						DataUnion.m_pByte[i*Col+nCol]=*(pData++);
				}
				break;
			case DATA_DOUBLE:
				{
					double* pData=srcMat.GetData<double>();
					for(int i=0;i<Row;i++)
						DataUnion.m_pByte[i*Col+nCol]=*(pData++);
				}
				break;
			case DATA_INT:
				{
					int* pData=srcMat.GetData<int>();
					for(int i=0;i<Row;i++)
						DataUnion.m_pByte[i*Col+nCol]=*(pData++);
				}
				break;
			}
		}
		break;
	case DATA_DOUBLE:
		{
			switch(srcMat.GetType())
			{
			case DATA_BYTE:
				{
					BYTE* pData=srcMat.GetData<BYTE>();
					for(int i=0;i<Row;i++)
						DataUnion.m_pDouble[i*Col+nCol]=*(pData++);
				}
				break;
			case DATA_DOUBLE:
				{
					double* pData=srcMat.GetData<double>();
					for(int i=0;i<Row;i++)
						DataUnion.m_pDouble[i*Col+nCol]=*(pData++);
				}
				break;
			case DATA_INT:
				{
					int* pData=srcMat.GetData<int>();
					for(int i=0;i<Row;i++)
						DataUnion.m_pDouble[i*Col+nCol]=*(pData++);
				}
				break;
			}
		}
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
	switch(m_dataType)
	{
	case DATA_DOUBLE:
		{
			double * pSrc=src.GetData<double>();
			for(int i=0;i<=rowE-rowS;i++)
			{
				for(int j=0;j<=colE-colS;j++)
				{
					for(int n=0;n<Depth;n++)
						DataUnion.m_pDouble[(i+rowS)*Col+j+colS+n]=pSrc[i*src.Col+j+n];
				}
			}
			break;
		}
	case DATA_BYTE:
		{
			BYTE * pSrc=src.GetData<BYTE>();
			for(int i=0;i<=rowE-rowS;i++)
			{
				for(int j=0;j<=colE-colS;j++)
				{
					for(int n=0;n<Depth;n++)
						DataUnion.m_pByte[(i+rowS)*Col+j+colS+n]=pSrc[i*src.Col+j+n];
				}
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
						DataUnion.m_pInt[(i+rowS)*Col+j+colS+n]=pSrc[i*src.Col+j+n];
				}
			}
			break;
		}
	}
	return PK_SUCCESS;
}


int CpkMat::getColData(CpkMat&dest,int nCol)
{
	if(nCol>=Col||nCol<0)
		return PK_NOT_ALLOW_OPERATOR;

	dest.Resize(Row,1,Depth,GetType());
	switch(m_dataType)
	{
	case DATA_BYTE:
		{
			BYTE* pBuff=dest.GetData<BYTE>();
			for(int i=0;i<Row;i++)
				*(pBuff++)=DataUnion.m_pByte[i*Col+nCol];
		}
		break;
	case DATA_DOUBLE:
		{
			double* pBuff=dest.GetData<double>();
			for(int i=0;i<Row;i++)
				*(pBuff++)=DataUnion.m_pDouble[i*Col+nCol];
		}
		break;
	case DATA_INT:
		{
			int* pBuff=dest.GetData<int>();
			for(int i=0;i<Row;i++)
				*(pBuff++)=DataUnion.m_pInt[i*Col+nCol];
		}
		break;
	}
	return PK_SUCCESS;
}

int CpkMat::getRow(CpkMat&dest,int nRow)
{
	if(nRow<0||nRow>=Row)
		return PK_NOT_ALLOW_OPERATOR;
	dest.Resize(1,Col,Depth,GetType());
	switch(m_dataType)
	{
	case DATA_BYTE:
		{
			BYTE* pBuff=dest.GetData<BYTE>();
			memcpy(pBuff,DataUnion.m_pByte+nRow*Col,sizeof(BYTE)*Col);
		}
		break;
	case DATA_DOUBLE:
		{
			double* pBuff=dest.GetData<double>();
			memcpy(pBuff,DataUnion.m_pDouble+nRow*Col,sizeof(double)*Col);
		}
		break;
	case DATA_INT:
		{
			int* pBuff=dest.GetData<int>();
			memcpy(pBuff,DataUnion.m_pInt+nRow*Col,sizeof(int)*Col);
		}
		break;
	}
	return PK_SUCCESS;
}

int CpkMat::GetData(CpkMat& dest,int rowS,int rowE,int colS,int colE)
{
	int arraySize=(rowE-rowS+1)*(colE-colS+1)*Depth;
	int dCol=colE-colS;
	if(colE>Col||rowE>Row||dCol==0)
		return PK_NOT_ALLOW_OPERATOR;

	switch(m_dataType)
	{
	case DATA_INT:
		{
			int * pdata=new int[arraySize];
			int * pSrc=DataUnion.m_pInt;
			for(int i=0;i<=rowE-rowS;i++)
			{
				for(int j=0;j<=colE-colS;j++)
				{
					for(int n=0;n<Depth;n++)
						pdata[i*dCol+j+n]=pSrc[(i+rowS)*Col+j+colS+n];
				}
			}
			dest.Resize(rowE-rowS+1,dCol,Depth,m_dataType,pdata);
			delete [] pdata;
			break;
		}
	case DATA_BYTE:
		{
			BYTE * pdata=new BYTE[arraySize];
			BYTE * pSrc=DataUnion.m_pByte;
			for(int i=0;i<=rowE-rowS;i++)
			{
				for(int j=0;j<=colE-colS;j++)
				{
					for(int n=0;n<Depth;n++)
						pdata[i*dCol+j+n]=pSrc[(i+rowS)*Col+j+colS+n];
				}
			}
			dest.Resize(rowE-rowS,dCol,Depth,m_dataType,pdata);
			delete [] pdata;
			break;
		}
	case DATA_DOUBLE:
		{
			if(dest.Row==0||dest.Col==0)
				dest.Resize(rowE-rowS+1,dCol,Depth,m_dataType);
			double * pSrc=DataUnion.m_pDouble;
			double* pdata=dest.GetData<double>();
			for(int i=0;i<=rowE-rowS;i++)
			{
				for(int j=0;j<=colE-colS;j++)
				{
					for(int n=0;n<Depth;n++)
						pdata[i*dCol+j+n]=pSrc[(i+rowS)*Col+j+colS+n];
				}
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