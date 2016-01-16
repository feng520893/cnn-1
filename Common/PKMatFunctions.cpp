#include "PKMatFunctions.h"
#include <fstream>
#include <ctime>

namespace pk
{
#ifndef max
#define max(a,b) (a>b)?(a):(b)
#endif

#ifndef min
#define min(a,b) (a>b)?(b):(a)
#endif



	int normalization(CpkMat&dest,CpkMat&src,double min,double max)
	{
		double tempMax=0,tempMin=99999,temp=0;
		BYTE* pByte=NULL;
		double* pDouble=NULL;
		int* pInt=NULL;
		switch (src.GetType())
		{
		case CpkMat::DATA_BYTE:
			{
				pByte=new BYTE[src.Depth*src.Row*src.Col];
				memcpy(pByte,src.GetData<BYTE>(),sizeof(BYTE)*src.Depth*src.Row*src.Col);
				for(int i=0;i<src.Depth*src.Row*src.Col;i++)
				{
					temp=pByte[i];
					if(temp>tempMax)
						tempMax=temp;
					if(temp<tempMin)
						tempMin=temp;
				}
				BYTE* buff=new BYTE[src.Depth*src.Row*src.Col];
				for(int i=0;i<src.Depth*src.Row*src.Col;i++)
					buff[i]=min+(pByte[i]-tempMin)/(tempMax-tempMin)*(max-min);
				dest.Resize(src.Row,src.Col,src.Depth,CpkMat::DATA_BYTE,buff);
				delete [] buff;
				buff=NULL;
				delete pByte;
				pByte=NULL;
			}
			break;
		case CpkMat::DATA_DOUBLE:
			{
				pDouble=new double[src.Depth*src.Row*src.Col];
				memcpy(pDouble,src.GetData<double>(),sizeof(double)*src.Depth*src.Row*src.Col);
				for(int i=0;i<src.Depth*src.Row*src.Col;i++)
				{
					temp=pDouble[i];
					if(temp>tempMax)
						tempMax=temp;
					if(temp<tempMin)
						tempMin=temp;
				}

				double* buff=new double[src.Depth*src.Row*src.Col];
				for(int i=0;i<src.Depth*src.Row*src.Col;i++)
					buff[i]=min+(pDouble[i]-tempMin)/(tempMax-tempMin)*(max-min);
				dest.Resize(src.Row,src.Col,src.Depth,CpkMat::DATA_DOUBLE,buff);
				delete [] buff;
				buff=NULL;
				delete pDouble;
				pDouble=NULL;
			}
			break;
		case CpkMat::DATA_INT:
			pInt=new int[src.Depth*src.Row*src.Col];
			memcpy(pInt,src.GetData<int>(),sizeof(int)*src.Depth*src.Row*src.Col);
			for(int i=0;i<src.Depth*src.Row*src.Col;i++)
			{
				temp=pInt[i];
				if(temp>tempMax)
					tempMax=temp;
				if(temp<tempMin)
					tempMin=temp;
			}
			int* buff=new int[src.Depth*src.Row*src.Col];
			for(int i=0;i<src.Depth*src.Row*src.Col;i++)
				buff[i]=min+(pDouble[i]-tempMin)/(tempMax-tempMin)*(max-min);
			dest.Resize(src.Row,src.Col,src.Depth,CpkMat::DATA_INT,buff);
			delete [] buff;
			buff=NULL;
			delete pInt;
			pInt=NULL;
			break;
		}
		return PK_SUCCESS;
	}

	double dot(CpkMat& x1,CpkMat& x2)
	{
		if(x1.GetType()!=CpkMat::DATA_DOUBLE||x2.GetType()!=CpkMat::DATA_DOUBLE)
			return PK_NOT_ALLOW_OPERATOR;
		if(x1.Row!=1||x2.Row!=1)
			return PK_NOT_ALLOW_OPERATOR;
		double ddot = 0;
		double* pX1=x1.GetData<double>();
		double* pX2=x2.GetData<double>();
		for(int i = 0; i <x1.Col; i++)
			ddot += pX1[i] * pX2[i];
		return ddot;
	}


	//////////////////////////////////////////////////////////////////////   
	// 内部函数，由SplitUV函数调用   
	//////////////////////////////////////////////////////////////////////   
	void ppp(double a[], double e[], double s[], double v[], int m, int n)   
	{    
		int i,j,p,q;   
		double d;   
		if (m>=n)    
			i=n;   
		else    
			i=m;   
		for (j=1; j<=i-1; j++)   
		{    
			a[(j-1)*n+j-1]=s[j-1];   
			a[(j-1)*n+j]=e[j-1];   
		}   
		a[(i-1)*n+i-1]=s[i-1];   
		if (m<n)    
			a[(i-1)*n+i]=e[i-1];   
		for (i=1; i<=n-1; i++)   
		{   
			for (j=i+1; j<=n; j++)   
			{    
				p=(i-1)*n+j-1;    
				q=(j-1)*n+i-1;   
				d=v[p];    
				v[p]=v[q];    
				v[q]=d;   
			}   
		}   
	}   

	//////////////////////////////////////////////////////////////////////   
	// 内部函数，由SplitUV函数调用   
	//////////////////////////////////////////////////////////////////////   
	void sss(double fg[2], double cs[2])   
	{    
		double r,d;   
		if ((fabs(fg[0])+fabs(fg[1]))==0.0)   
		{    
			cs[0]=1.0;    
			cs[1]=0.0;    
			d=0.0;   
		}   
		else    
		{    
			d=sqrt(fg[0]*fg[0]+fg[1]*fg[1]);   
			if (fabs(fg[0])>fabs(fg[1]))   
			{    
				d=fabs(d);   
				if (fg[0]<0.0)    
					d=-d;   
			}   
			if (fabs(fg[1])>=fabs(fg[0]))   
			{    
				d=fabs(d);   
				if (fg[1]<0.0)    
					d=-d;   
			}   

			cs[0]=fg[0]/d;    
			cs[1]=fg[1]/d;   
		}   
		r=1.0;   
		if (fabs(fg[0])>fabs(fg[1]))    
			r=cs[1];   
		else if (cs[0]!=0.0)    
			r=1.0/cs[0];   

		fg[0]=d;    
		fg[1]=r;   
	} 

	CpkMat diag(CpkMat&src)
	{
		CpkMat tmp;
		if(src.Col!=1&&src.Col!=1)
			return tmp;
		int length=max(src.Col,src.Row);
		tmp.Resize(length,length,1,src.GetType());
		switch(src.GetType())
		{
		case CpkMat::DATA_DOUBLE:
			for(int i=0;i<length;i++)
				tmp.GetData<double>()[i*tmp.lineSize+i]=src.GetData<double>()[i];
			break;
		case CpkMat::DATA_BYTE:
			for(int i=0;i<length;i++)
				tmp.GetData<BYTE>()[i*tmp.lineSize+i]=src.GetData<BYTE>()[i];
			break;
		case CpkMat::DATA_INT:
			for(int i=0;i<length;i++)
				tmp.GetData<int>()[i*tmp.lineSize+i]=src.GetData<int>()[i];
			break;
		}
		return tmp;
	}

	int svd(CpkMat& src,CpkMat& mtxU,CpkMat& mtxV,CpkMat& mtxS,double eps/*= 0.000001*/)
	{    
		int i,j,k,l,it,ll,kk,ix,iy,mm,nn,iz,m1,ks;   
		double d,dd,t,sm,sm1,em1,sk,ek,b,c,shh,fg[2],cs[2];   
		double *s,*e,*w;   
		int m = src.Row;   
		int n = src.Col;   
		// 初始化U, V矩阵   
		if (mtxU.Resize(m, m,1,CpkMat::DATA_DOUBLE)!=PK_SUCCESS || mtxV.Resize(n, n,1,CpkMat::DATA_DOUBLE)!=PK_SUCCESS)   
			return false;   
		mtxS=src;
		// 临时缓冲区   
		int ka = max(m, n) + 1;   
		s = new double[ka]; 
		memset(s,0,sizeof(double)*ka);
		e = new double[ka];   
		memset(e,0,sizeof(double)*ka);
		w = new double[ka];
		memset(w,0,sizeof(double)*ka);
		// 指定迭代次数为60   
		double* m_pData=mtxS.GetData<double>();
		double* uPData=mtxU.GetData<double>();
		double* vPData=mtxV.GetData<double>();
		it=60;    
		k=n;   
		if (m-1<n)    
			k=m-1;   
		l=m;   
		if (n-2<m)    
			l=n-2;   
		if (l<0)    
			l=0;   
		// 循环迭代计算   
		ll=k;   
		if (l>k)    
			ll=l;   
		if (ll>=1)   
		{    
			for (kk=1; kk<=ll; kk++)   
			{    
				if (kk<=k)   
				{    
					d=0.0;   
					for (i=kk; i<=m; i++)   
					{    
						ix=(i-1)*n+kk-1;    
						d=d+m_pData[ix]*m_pData[ix];   
					}   
					s[kk-1]=sqrt(d);   
					if (s[kk-1]!=0.0)   
					{    
						ix=(kk-1)*n+kk-1;   
						if (m_pData[ix]!=0.0)   
						{    
							s[kk-1]=fabs(s[kk-1]);   
							if (m_pData[ix]<0.0)    
								s[kk-1]=-s[kk-1];   
						}   
						for (i=kk; i<=m; i++)   
						{    
							iy=(i-1)*n+kk-1;   
							m_pData[iy]=m_pData[iy]/s[kk-1];   
						}   
						m_pData[ix]=1.0+m_pData[ix];   
					}   
					s[kk-1]=-s[kk-1];   
				}   
				if (n>=kk+1)   
				{    
					for (j=kk+1; j<=n; j++)   
					{    
						if ((kk<=k)&&(s[kk-1]!=0.0))   
						{    
							d=0.0;   
							for (i=kk; i<=m; i++)   
							{    
								ix=(i-1)*n+kk-1;   
								iy=(i-1)*n+j-1;   
								d=d+m_pData[ix]*m_pData[iy];   
							}   
							d=-d/m_pData[(kk-1)*n+kk-1];   
							for (i=kk; i<=m; i++)   
							{    
								ix=(i-1)*n+j-1;   
								iy=(i-1)*n+kk-1;   
								if(kk>650)
									kk=kk;
								m_pData[ix]=m_pData[ix]+d*m_pData[iy];   
							}  
						}   
						e[j-1]=m_pData[(kk-1)*n+j-1];   
					}   
				}   
				if (kk<=k)   
				{    
					for (i=kk; i<=m; i++)   
					{    
						ix=(i-1)*m+kk-1;    
						iy=(i-1)*n+kk-1;   
						uPData[ix]=m_pData[iy];   
					}   
				}   
				if (kk<=l)   
				{    
					d=0.0;   
					for (i=kk+1; i<=n; i++)   
						d=d+e[i-1]*e[i-1];   
					e[kk-1]=sqrt(d);   
					if (e[kk-1]!=0.0)   
					{    
						if (e[kk]!=0.0)   
						{    
							e[kk-1]=fabs(e[kk-1]);   
							if (e[kk]<0.0)    
								e[kk-1]=-e[kk-1];   
						}   
						for (i=kk+1; i<=n; i++)   
							e[i-1]=e[i-1]/e[kk-1];   
						e[kk]=1.0+e[kk];   
					}   
					e[kk-1]=-e[kk-1];   
					if ((kk+1<=m)&&(e[kk-1]!=0.0))   
					{    
						for (i=kk+1; i<=m; i++)    
							w[i-1]=0.0;   
						for (j=kk+1; j<=n; j++)   
							for (i=kk+1; i<=m; i++)   
								w[i-1]=w[i-1]+e[j-1]*m_pData[(i-1)*n+j-1];   
						for (j=kk+1; j<=n; j++)   
						{   
							for (i=kk+1; i<=m; i++)   
							{    
								ix=(i-1)*n+j-1;   
								m_pData[ix]=m_pData[ix]-w[i-1]*e[j-1]/e[kk];   
							}   
						}   
					}   
					for (i=kk+1; i<=n; i++)   
						vPData[(i-1)*n+kk-1]=e[i-1];   
				}   
			}   
		}   
		mm=n;   
		if (m+1<n)    
			mm=m+1;   
		if (k<n)    
			s[k]=m_pData[k*n+k];   
		if (m<mm)    
			s[mm-1]=0.0;   
		if (l+1<mm)    
			e[l]=m_pData[l*n+mm-1];   
		e[mm-1]=0.0;   
		nn=m;   
		if (m>n)    
			nn=n;   
		if (nn>=k+1)   
		{    
			for (j=k+1; j<=nn; j++)   
			{    
				for (i=1; i<=m; i++)   
					uPData[(i-1)*m+j-1]=0.0;   
				uPData[(j-1)*m+j-1]=1.0;   
			}   
		}   
		if (k>=1)   
		{    
			for (ll=1; ll<=k; ll++)   
			{    
				kk=k-ll+1;    
				iz=(kk-1)*m+kk-1;   
				if (s[kk-1]!=0.0)   
				{    
					if (nn>=kk+1)   
					{   
						for (j=kk+1; j<=nn; j++)   
						{    
							d=0.0;   
							for (i=kk; i<=m; i++)   
							{    
								ix=(i-1)*m+kk-1;   
								iy=(i-1)*m+j-1;   
								d=d+uPData[ix]*uPData[iy]/uPData[iz];   
							}   
							d=-d;   
							for (i=kk; i<=m; i++)   
							{    
								ix=(i-1)*m+j-1;   
								iy=(i-1)*m+kk-1;   
								uPData[ix]=uPData[ix]+d*uPData[iy];   
							}   
						}   
					}   
					for (i=kk; i<=m; i++)   
					{    
						ix=(i-1)*m+kk-1;    
						uPData[ix]=-uPData[ix];   
					}   
					uPData[iz]=1.0+uPData[iz];   
					if (kk-1>=1)   
					{   
						for (i=1; i<=kk-1; i++)   
							uPData[(i-1)*m+kk-1]=0.0;   
					}   
				}   
				else   
				{    
					for (i=1; i<=m; i++)   
						uPData[(i-1)*m+kk-1]=0.0;   
					uPData[(kk-1)*m+kk-1]=1.0;   
				}   
			}   
		}   
		for (ll=1; ll<=n; ll++)   
		{    
			kk=n-ll+1;    
			iz=kk*n+kk-1;   
			if ((kk<=l)&&(e[kk-1]!=0.0))   
			{    
				for (j=kk+1; j<=n; j++)   
				{    
					d=0.0;   
					for (i=kk+1; i<=n; i++)   
					{    
						ix=(i-1)*n+kk-1;    
						iy=(i-1)*n+j-1;   
						d=d+vPData[ix]*vPData[iy]/vPData[iz];   
					}   
					d=-d;   
					for (i=kk+1; i<=n; i++)   
					{    
						ix=(i-1)*n+j-1;    
						iy=(i-1)*n+kk-1;   
						vPData[ix]=vPData[ix]+d*vPData[iy];   
					}   
				}   
			}   
			for (i=1; i<=n; i++)   
				vPData[(i-1)*n+kk-1]=0.0;   
			vPData[iz-n]=1.0;   
		}   
		for (i=1; i<=m; i++)   
			for (j=1; j<=n; j++)   
				m_pData[(i-1)*n+j-1]=0.0;   
		m1=mm;    
		it=60;   
		while (1)   
		{    
			if (mm==0)   
			{    
				ppp(m_pData,e,s,vPData,m,n);   
				return PK_SUCCESS;   
			}   
			if (it==0)   
			{    
				ppp(m_pData,e,s,vPData,m,n);   
				return PK_FAIL;   
			}   
			kk=mm-1;   
			while ((kk!=0)&&(fabs(e[kk-1])!=0.0))   
			{    
				d=fabs(s[kk-1])+fabs(s[kk]);   
				dd=fabs(e[kk-1]);   
				if (dd>eps*d)    
					kk=kk-1;   
				else    
					e[kk-1]=0.0;   
			}   
			if (kk==mm-1)   
			{    
				kk=kk+1;   
				if (s[kk-1]<0.0)   
				{    
					s[kk-1]=-s[kk-1];   
					for (i=1; i<=n; i++)   
					{    
						ix=(i-1)*n+kk-1;    
						vPData[ix]=-vPData[ix];}   
				}   
				while ((kk!=m1)&&(s[kk-1]<s[kk]))   
				{    
					d=s[kk-1];    
					s[kk-1]=s[kk];    
					s[kk]=d;   
					if (kk<n)   
					{   
						for (i=1; i<=n; i++)   
						{    
							ix=(i-1)*n+kk-1;    
							iy=(i-1)*n+kk;   
							d=vPData[ix];    
							vPData[ix]=vPData[iy];    
							vPData[iy]=d;   
						}   
					}   
					if (kk<m)   
					{   
						for (i=1; i<=m; i++)   
						{    
							ix=(i-1)*m+kk-1;    
							iy=(i-1)*m+kk;   
							d=uPData[ix];    
							uPData[ix]=uPData[iy];    
							uPData[iy]=d;   
						}   
					}   
					kk=kk+1;   
				}   
				it=60;   
				mm=mm-1;   
			}   
			else   
			{    
				ks=mm;   
				while ((ks>kk)&&(fabs(s[ks-1])!=0.0))   
				{    
					d=0.0;   
					if (ks!=mm)    
						d=d+fabs(e[ks-1]);   
					if (ks!=kk+1)    
						d=d+fabs(e[ks-2]);   
					dd=fabs(s[ks-1]);   
					if (dd>eps*d)    
						ks=ks-1;   
					else    
						s[ks-1]=0.0;   
				}   
				if (ks==kk)   
				{    
					kk=kk+1;   
					d=fabs(s[mm-1]);   
					t=fabs(s[mm-2]);   
					if (t>d)    
						d=t;   
					t=fabs(e[mm-2]);   
					if (t>d)    
						d=t;   
					t=fabs(s[kk-1]);   
					if (t>d)    
						d=t;   
					t=fabs(e[kk-1]);   
					if (t>d)    
						d=t;   
					sm=s[mm-1]/d;    
					sm1=s[mm-2]/d;   
					em1=e[mm-2]/d;   
					sk=s[kk-1]/d;    
					ek=e[kk-1]/d;   
					b=((sm1+sm)*(sm1-sm)+em1*em1)/2.0;   
					c=sm*em1;    
					c=c*c;    
					shh=0.0;   

					if ((b!=0.0)||(c!=0.0))   
					{    
						shh=sqrt(b*b+c);   
						if (b<0.0)    
							shh=-shh;   
						shh=c/(b+shh);   
					}   
					fg[0]=(sk+sm)*(sk-sm)-shh;   
					fg[1]=sk*ek;   
					for (i=kk; i<=mm-1; i++)   
					{    
						sss(fg,cs);   
						if (i!=kk)    
							e[i-2]=fg[0];   
						fg[0]=cs[0]*s[i-1]+cs[1]*e[i-1];   
						e[i-1]=cs[0]*e[i-1]-cs[1]*s[i-1];   
						fg[1]=cs[1]*s[i];   
						s[i]=cs[0]*s[i];   
						if ((cs[0]!=1.0)||(cs[1]!=0.0))   
						{   
							for (j=1; j<=n; j++)   
							{    
								ix=(j-1)*n+i-1;   
								iy=(j-1)*n+i;   

								d=cs[0]*vPData[ix]+cs[1]*vPData[iy];   

								vPData[iy]=-cs[1]*vPData[ix]+cs[0]*vPData[iy];   
								vPData[ix]=d;   
							}   
						}   
						sss(fg,cs);   
						s[i-1]=fg[0];   
						fg[0]=cs[0]*e[i-1]+cs[1]*s[i];   
						s[i]=-cs[1]*e[i-1]+cs[0]*s[i];   
						fg[1]=cs[1]*e[i];   
						e[i]=cs[0]*e[i];   
						if (i<m)   
						{   
							if    
								((cs[0]!=1.0)||(cs[1]!=0.0))   
							{   
								for (j=1; j<=m; j++)   
								{    
									ix=(j-1)*m+i-1;   
									iy=(j-1)*m+i;   

									d=cs[0]*uPData[ix]+cs[1]*uPData[iy];   

									uPData[iy]=-cs[1]*uPData[ix]+cs[0]*uPData[iy];   

									uPData[ix]=d;   
								}   
							}   
						}   
					}   

					e[mm-2]=fg[0];   
					it=it-1;   
				}   
				else   
				{    
					if (ks==mm)   
					{    
						kk=kk+1;   
						fg[1]=e[mm-2];    
						e[mm-2]=0.0;   
						for (ll=kk; ll<=mm-1; ll++)   
						{    
							i=mm+kk-ll-1;   
							fg[0]=s[i-1];   
							sss(fg,cs);   
							s[i-1]=fg[0];   
							if (i!=kk)   
							{    
								fg[1]=-cs[1]*e[i-2];   
								e[i-2]=cs[0]*e[i-2];   
							}   
							if    
								((cs[0]!=1.0)||(cs[1]!=0.0))   
							{   
								for (j=1; j<=n; j++)   
								{    
									ix=(j-1)*n+i-1;   

									iy=(j-1)*n+mm-1;   

									d=cs[0]*vPData[ix]+cs[1]*vPData[iy];   

									vPData[iy]=-cs[1]*vPData[ix]+cs[0]*vPData[iy];   

									vPData[ix]=d;   
								}   
							}   
						}   
					}   
					else   
					{    
						kk=ks+1;   
						fg[1]=e[kk-2];   
						e[kk-2]=0.0;   
						for (i=kk; i<=mm; i++)   
						{    
							fg[0]=s[i-1];   
							sss(fg,cs);   
							s[i-1]=fg[0];   
							fg[1]=-cs[1]*e[i-1];   
							e[i-1]=cs[0]*e[i-1];   
							if ((cs[0]!=1.0)||(cs[1]!=0.0))   
							{   
								for (j=1; j<=m; j++)   
								{    

									ix=(j-1)*m+i-1;   

									iy=(j-1)*m+kk-2;   

									d=cs[0]*uPData[ix]+cs[1]*uPData[iy];   

									uPData[iy]=-cs[1]*uPData[ix]+cs[0]*uPData[iy];   

									uPData[ix]=d;   
								}   
							}   
						}   
					}   
				}   
			}   
		}   

		delete [] s;   
		delete [] e;   
		delete [] w;   
		return PK_SUCCESS;   
	}


	int avg(CpkMat&dest,CpkMat&src,DATA_TYPE dataType)
	{
		if(dataType==DATA_COLS)
		{
			dest.Resize(1,src.Col,1,src.GetType());
			switch(src.GetType())
			{
			case CpkMat::DATA_DOUBLE:
				{
					double* buffer=dest.GetData<double>();
					double* pData=src.GetData<double>();
					for(int i=0;i<src.Col;i++)
					{	
						for(int j=0;j<src.Row;j++)
							buffer[i]+=pData[j*src.Col+i];
						buffer[i]/=src.Row;
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
						buffer[i]/=src.Row;
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
						buffer[i]/=src.Row;
					}
				}
			}
		}
		else
		{
			dest.Resize(src.Row,1,1,src.GetType());
			switch(src.GetType())
			{
			case CpkMat::DATA_DOUBLE:
				{
					double* buffer=dest.GetData<double>();
					double* pData=src.GetData<double>();
					for(int i=0;i<src.Row;i++)
					{	
						for(int j=0;j<src.Col;j++)
							buffer[i]+=pData[i*src.Col+j];
						buffer[i]/=src.Col;
					}
				}
				break;
				case CpkMat::DATA_INT:
				{
					int* buffer=dest.GetData<int>();
					int* pData=src.GetData<int>();
					for(int i=0;i<src.Row;i++)
					{	
						for(int j=0;j<src.Col;j++)
							buffer[i]+=pData[i*src.Col+j];
						buffer[i]/=src.Col;
					}
				}
				break;
				case CpkMat::DATA_BYTE:
				{
					BYTE* buffer=dest.GetData<BYTE>();
					BYTE* pData=src.GetData<BYTE>();
					for(int i=0;i<src.Row;i++)
					{	
						for(int j=0;j<src.Col;j++)
							buffer[i]+=pData[i*src.Col+j];
						buffer[i]/=src.Col;
					}
				}
				break;
			}
		}
		return PK_SUCCESS;
	}

	//公式a[i][j] = b[n - i - 1][n - j - 1],a为变化前矩阵,下标从0开始
	int rot180(CpkMat&dest,CpkMat&src)
	{
		CpkMat tmp;
		tmp.Resize(src.Row,src.Col,src.Depth,src.GetType());
		switch(src.GetType())
		{
		case CpkMat::DATA_INT:
		{
			int *pSdata=src.GetData<int>();
			int *pData=tmp.GetData<int>();
			for(int i=0;i<src.Row;i++)
				for(int j=0;j<src.Col;j++)
					pData[(src.Row-i-1)*src.Col+src.Col-j-1]=pSdata[i*src.Col+j];
			break;
		}
		case CpkMat::DATA_BYTE:
			{
				BYTE *pSdata=src.GetData<BYTE>();
				BYTE *pData=tmp.GetData<BYTE>();
				for(int i=0;i<src.Row;i++)
					for(int j=0;j<src.Col;j++)
						pData[(src.Row-i-1)*src.Col+src.Col-j-1]=pSdata[i*src.Col+j];
				break;
			}
		case CpkMat::DATA_DOUBLE:
			{
				double *pSdata=src.GetData<double>();
				double *pData=tmp.GetData<double>();
				for(int i=0;i<src.Row;i++)
					for(int j=0;j<src.Col;j++)
						pData[(src.Row-i-1)*src.Col+src.Col-j-1]=pSdata[i*src.Col+j];
				break;
			}
		}
		tmp.copyTo(dest,tmp.GetType());
		return PK_SUCCESS;
	}

	int conv2(CpkMat& dest,CpkMat&src,CpkMat&mask,bool bFull)
	{
		int nRet=PK_SUCCESS;
		if(!bFull)
		{
			if(src.Row<=mask.Row&&src.Col<=mask.Col)
				return PK_NOT_ALLOW_OPERATOR;
			int dRow=src.Row-mask.Row+1;
			int dCol=src.Col-mask.Col+1;
			dest.Resize(dRow,dCol,1,CpkMat::DATA_DOUBLE);
			double* pData=dest.GetData<double>();
			CpkMat rowMat;
			mask.RowVector(rowMat);
			rowMat=rowMat.Transpose();
			for(int i=0;i<dRow;i++)
			{
				for(int j=0;j<dCol;j++)
				{
					CpkMat tmp;
					src.GetData(tmp,i,i+mask.Row-1,j,j+mask.Col-1);
					tmp.RowVector();
					tmp=tmp*rowMat;
					pData[i*dest.Col+j]=*(tmp.GetData<double>());
				}
			}
		}
		else
		{
// 			int dRow=src.Row+(mask.Row-1)*2;
// 			int dCol=src.Col+(mask.Col-1)*2;
// 			CpkMat tmp;
// 			pk::rot180(mask,mask);
// 			tmp.Resize(dRow,dCol,1,src.GetType());
// 			tmp.setData(src,mask.Row-1,src.Row,mask.Col-1,src.Col);
// 			nRet=conv2(dest,tmp,mask,false);

			int src_row = src.Row;
			int kernel_row = mask.Row;
			int src_cols = src.Col;
			int kernel_cols = mask.Col;
			int dst_row = src_row + kernel_row - 1;  
			int dst_cols = src_cols + kernel_cols - 1;        
			int edge_row = kernel_row - 1;  
			int edge_cols = kernel_cols - 1;  
			int kernel_i=0,src_i=0,kernel_j=0,src_j=0;

			dest.Resize(dst_row,dst_cols,1,CpkMat::DATA_DOUBLE);
			double* pDest=dest.GetData<double>();
			double* pSrc=src.GetData<double>();
			double* pMask=mask.GetData<double>();
			for (int i = 0; i < dst_row; i++) 
			{    
				for (int j = 0; j < dst_cols; j++) 
				{       
					double sum = 0;  
					kernel_i = kernel_row - 1 - max(0, edge_row - i);  
					src_i = max(0, i - edge_row);  
					for (; kernel_i >= 0 && src_i < src_row; kernel_i--, src_i++)
					{  
						kernel_j = kernel_cols - 1 - max(0, edge_cols - j);  
						src_j =max(0, j - edge_cols);  
						for (; kernel_j >= 0 && src_j < src_cols; kernel_j--, src_j++)
							sum += pSrc[src_i*src.Col+src_j] * pMask[kernel_i*mask.Col+kernel_j];  
					}             
					pDest[i*dst_cols+j] = sum;  
				}  
			}  
		}
		return nRet;
	}

	CpkMat exp(CpkMat&src)
	{
		CpkMat temp(src.Row,src.Col,src.Depth,src.GetType());
		switch(src.GetType())
		{
		case CpkMat::DATA_DOUBLE:
			{
				double* pSrc=src.GetData<double>();
				double* pBuff=temp.GetData<double>();
				for(int i=0;i<src.Row;i++)
					for(int j=0;j<src.Col;j++)
						pBuff[i*src.Col+j]=::exp(pSrc[i*src.Col+j]);
			}
			break;
		}
		return temp;
	}

	CpkMat log(CpkMat&src)
	{
		CpkMat temp(src.Row,src.Col,src.Depth,src.GetType());
		switch(src.GetType())
		{
		case CpkMat::DATA_DOUBLE:
			{
				double* pSrc=src.GetData<double>();
				double* pBuff=temp.GetData<double>();
				for(int i=0;i<src.Row;i++)
					for(int j=0;j<src.Col;j++)
						pBuff[i*src.Col+j]=::log(pSrc[i*src.Col+j]);
			}
			break;
		}
		return temp;
	}

	CpkMat pow(CpkMat&src,int num)
	{
		CpkMat temp(src.Row,src.Col,src.Depth,src.GetType());
		switch(src.GetType())
		{
		case CpkMat::DATA_DOUBLE:
			{
				double* pSrc=src.GetData<double>();
				double* pBuff=temp.GetData<double>();
				for(int i=0;i<src.Row;i++)
					for(int j=0;j<src.Col;j++)
						pBuff[i*src.Col+j]=::pow(pSrc[i*src.Col+j],num);
			}
			break;
		}
		return temp;
	}

	CpkMat rand(int row,int col)
	{
		srand(time(NULL));
		CpkMat temp(row,col,1,CpkMat::DATA_DOUBLE);
		double* pData=temp.GetData<double>();
		for(int i=0;i<row;i++)
			for(int j=0;j<col;j++)
				pData[i*col+j]=(double)::rand()/(double)RAND_MAX;
		return temp;
	}

	CpkMat randn(int row,int col,float zoomSize)
	{
/*		srand(time(NULL));
		CpkMat temp(row,col,1,CpkMat::DATA_DOUBLE);
		float x1,x2;
		double* pData=temp.GetData<double>();
		for(int i=0;i<row;i++)
		{
			for(int j=0;j<col;j++)
			{
				//+0.0005防止出现0导致,log(0)未定义
				x1=::rand()/(double)RAND_MAX+0.00005;
				x2=::rand()/(double)RAND_MAX+0.00005;
				pData[i*col+j]=::sqrt(-2*::log(x1))*::cos(x2*M_PI);
			}
		}
		return temp;*/
		srand(time(NULL));
		CpkMat temp(row,col,1,CpkMat::DATA_DOUBLE);
		double* pData=temp.GetData<double>();
		for(int i=0;i<row;i++)
		{
			for(int j=0;j<col;j++)
			{
				static double V1, V2, S;
				static int phase = 0;
				double X;
				if ( phase == 0 ) 
				{
					do {
						double U1 = (double)::rand() / RAND_MAX;
						double U2 = (double)::rand() / RAND_MAX;
						V1 = 2 * U1 - 1;
						V2 = 2 * U2 - 1;
						S = V1 * V1 + V2 * V2;
					} while(S >= 1 || S == 0);
					X = V1 * sqrt(-2 * ::log(S) / S);
				} else
					X = V2 * sqrt(-2 * ::log(S) / S);
				phase = 1 - phase;
				pData[i*col+j]=X*zoomSize;
			}
		}
		return temp;
	}

	CpkMat randperm(int num)
	{
		int* tmpNum=new int[num];
		for(int i=0;i<num;i++)
			tmpNum[i]=i;
		srand(time(NULL));
		int nPos,nTemp;
		for(int i=0;i<num;i++)
		{
			nPos=::rand()%(num-1);
			nTemp=tmpNum[i];
			tmpNum[i]=tmpNum[nPos];
			tmpNum[nPos]=nTemp;
		}
		CpkMat tmp(1,num,1,CpkMat::DATA_INT,tmpNum);
		return tmp;
	}


	TwoValue<double> Max(CpkMat&src)
	{
		CpkMat tmp;
		src.copyTo(tmp,src.GetType());
		tmp.ColumnVector();
		TwoValue<double> t;
		t.x1=t.x2=0;
		switch(tmp.GetType())
		{
		case CpkMat::DATA_DOUBLE:
			{
				double* pData=tmp.GetData<double>();
				for(int i=0;i<tmp.Row*tmp.Col;i++)
				{
					if(t.x1<pData[i])
					{
						t.x1=pData[i];
						t.x2=i;
					}
				}
			}
			break;
		case CpkMat::DATA_INT:
			{
				int* pData=tmp.GetData<int>();
				for(int i=0;i<tmp.Col;i++)
				{
					if(t.x1<pData[i])
					{
						t.x1=pData[i];
						t.x2=i;
					}
				}
			}
			break;
		case CpkMat::DATA_BYTE:
			{
				BYTE* pData=tmp.GetData<BYTE>();
				for(int i=0;i<tmp.Col;i++)
				{
					if(t.x1<pData[i])
					{
						t.x1=pData[i];
						t.x2=i;
					}
				}
			}
			break;
		}
		return t;
	}

	TwoValue<CpkMat> Max(CpkMat&src,int type)
	{
		TwoValue<CpkMat> tmp;
		if(type==1)
		{
			tmp.x1.Resize(1,src.Col,1,CpkMat::DATA_DOUBLE);
			tmp.x2.Resize(1,src.Col,1,CpkMat::DATA_INT);
			double* pSrc=src.GetData<double>();
			double* pX1=tmp.x1.GetData<double>();
			int* pX2=tmp.x2.GetData<int>();
			for(int i=0;i<src.Col;i++)
			{
				double dTmp=0;
				int index=0;
				for(int j=0;j<src.Row;j++)
				{
					if(pSrc[j*src.Col+i]>dTmp)
					{
						dTmp=pSrc[j*src.Col+i];
						index=j;
					}
				}
				pX1[i]=dTmp;
				pX2[i]=index;
			}
		}
		else
		{
			tmp.x1.Resize(src.Row,1,1,CpkMat::DATA_DOUBLE);
			tmp.x2.Resize(src.Row,1,1,CpkMat::DATA_DOUBLE);
			double* pSrc=src.GetData<double>();
			double* pX1=tmp.x1.GetData<double>();
			double* pX2=tmp.x2.GetData<double>();
			for(int i=0;i<src.Row;i++)
			{
				double dTmp=0;
				int index=0;
				for(int j=0;j<src.Col;j++)
				{
					if(pSrc[i*src.Col+j]>dTmp)
					{
						dTmp=pSrc[j*src.Row+i];
						index=j;
					}
				}
				pX1[i]=dTmp;
				pX2[i]=index;
			}
		}
		return tmp;
	}

	CpkMat sum(CpkMat&src, int type)
	{
		CpkMat tmp;
		if(type==1)
		{
			tmp.Resize(1,src.Col,1,CpkMat::DATA_DOUBLE);
			double* pSrc=src.GetData<double>();
			double* pX1=tmp.GetData<double>();
			for(int i=0;i<src.Col;i++)
			{
				double sum=0;
				for(int j=0;j<src.Row;j++)
					sum+=pSrc[j*src.Col+i];
				pX1[i]=sum;
			}
		}
		else
		{
			tmp.Resize(src.Row,1,1,CpkMat::DATA_DOUBLE);
			double* pSrc=src.GetData<double>();
			double* pX1=tmp.GetData<double>();
			for(int i=0;i<src.Row;i++)
			{
				double sum=0;
				for(int j=0;j<src.Col;j++)
					sum+=pSrc[i*src.Col+j];
				pX1[i]=sum;
			}
		}
		return tmp;
	}

	CpkMat reshape(CpkMat&src,int row,int col)
	{
		CpkMat tmp;
		if(src.Col*src.Row!=row*col)
			return tmp;
		switch(src.GetType())
		{
		case CpkMat::DATA_DOUBLE:
			{
				tmp.Resize(row,col,1,CpkMat::DATA_DOUBLE);
				double *pSdata=src.GetData<double>();
				double *pData=tmp.GetData<double>();
					for(int i=0;i<col;i++)
						for(int j=0;j<row;j++)
							pData[j*col+i]=*pSdata++;
			}
			break;
		case CpkMat::DATA_INT:
			{
				tmp.Resize(row,col,1,CpkMat::DATA_INT);
				int *pSdata=src.GetData<int>();
				int *pData=tmp.GetData<int>();
				for(int i=0;i<col;i++)
					for(int j=0;j<row;j++)
						pData[j*col+i]=*pSdata++;
			}
			break;
		case CpkMat::DATA_BYTE:
			{
				tmp.Resize(row,col,1,CpkMat::DATA_BYTE);
				BYTE *pSdata=src.GetData<BYTE>();
				BYTE *pData=tmp.GetData<BYTE>();
				for(int i=0;i<col;i++)
					for(int j=0;j<row;j++)
						pData[j*col+i]=*pSdata++;
			}
			break;
		}
		return tmp;
	}

	CpkMat kron(CpkMat&src1,CpkMat&src2)
	{
		CpkMat tmp(src1.Row*src2.Row,src1.Col*src2.Col,1,CpkMat::DATA_DOUBLE);
		double* pTmp=tmp.GetData<double>();
		double* pSrc1=src1.GetData<double>();
		double* pSrc2=src2.GetData<double>();
		for(int i=0;i<src1.Row;i++)
		{
			for(int j=0;j<src1.Col;j++)
			{
				for(int k=0;k<src2.Row;k++)
				{
					for(int g=0;g<src2.Col;g++)
					{
						int offsetR=i*src2.Row+k;
						int offfsetC=j*src2.Col+g;
						pTmp[offsetR*tmp.Col+offfsetC]=pSrc1[i*src1.Col+j]*pSrc2[k*src2.Col+g];
					}
				}
			}
		}
		return tmp;
	}

	CpkMat subVec(CpkMat&src,CpkMat&vec,DATA_TYPE dataType)
	{
		CpkMat tmp;
		if((dataType==DATA_ROWS&&vec.Row!=1||vec.Col!=src.Col)&&(dataType==DATA_COLS&&vec.Col!=1||vec.Row!=src.Row))
			return tmp;
		tmp.Resize(src.Row,src.Col,1,src.GetType());
		if(dataType==DATA_COLS)
		{
			switch(src.GetType())
			{
			case CpkMat::DATA_DOUBLE:
				{
					double* pSrc=src.GetData<double>();
					double* pVec=vec.GetData<double>();
					double* pDest=tmp.GetData<double>(); 
					for(int i=0;i<src.Row;i++)
					{
						for(int j=0;j<src.Col;j++)
							pDest[i*tmp.lineSize+j]=pSrc[i*src.lineSize+j]-pVec[i];
					}
				}
				break;
				case CpkMat::DATA_INT:
				{
					int* pSrc=src.GetData<int>();
					int* pVec=vec.GetData<int>();
					int* pDest=tmp.GetData<int>(); 
					for(int i=0;i<src.Row;i++)
					{
						for(int j=0;j<src.Col;j++)
							pDest[i*tmp.lineSize+j]=pSrc[i*src.lineSize+j]-pVec[i];
					}
				}
				break;
				case CpkMat::DATA_BYTE:
				{
					BYTE* pSrc=src.GetData<BYTE>();
					BYTE* pVec=vec.GetData<BYTE>();
					BYTE* pDest=tmp.GetData<BYTE>(); 
					for(int i=0;i<src.Row;i++)
					{
						for(int j=0;j<src.Col;j++)
							pDest[i*tmp.lineSize+j]=pSrc[i*src.lineSize+j]-pVec[i];
					}
				}
				break;
			}
		}
		else
		{
			switch(src.GetType())
			{
			case CpkMat::DATA_DOUBLE:
				{
					double* pSrc=src.GetData<double>();
					double* pVec=vec.GetData<double>();
					double* pDest=tmp.GetData<double>(); 
					for(int i=0;i<src.Col;i++)
					{
						for(int j=0;j<src.Row;j++)
							pDest[j*tmp.lineSize+i]=pSrc[j*src.lineSize+i]-pVec[i];
					}
				}
				break;
			case CpkMat::DATA_INT:
				{
					int* pSrc=src.GetData<int>();
					int* pVec=vec.GetData<int>();
					int* pDest=tmp.GetData<int>(); 
					for(int i=0;i<src.Col;i++)
					{
						for(int j=0;j<src.Row;j++)
							pDest[j*tmp.lineSize+i]=pSrc[j*src.lineSize+i]-pVec[i];
					}
				}
				break;
			case CpkMat::DATA_BYTE:
				{
					BYTE* pSrc=src.GetData<BYTE>();
					BYTE* pVec=vec.GetData<BYTE>();
					BYTE* pDest=tmp.GetData<BYTE>(); 
					for(int i=0;i<src.Col;i++)
					{
						for(int j=0;j<src.Row;j++)
							pDest[j*tmp.lineSize+i]=pSrc[j*src.lineSize+i]-pVec[i];
					}
				}
				break;
			}
		}
		return tmp;
	}

	CpkMat sub(double num,CpkMat&src)
	{
		CpkMat tmp(src.Row,src.Col,1,src.GetType());
		double* pTmp=tmp.GetData<double>();
		double* pSrc=src.GetData<double>();
		for(int i=0;i<src.Row*src.Col;i++)
			pTmp[i]=num-pSrc[i];
		return tmp;
	}

	CpkMat matDotMat(CpkMat& x1,CpkMat& x2)
	{
		CpkMat tmp(x1.Row,x1.Col,1,x1.GetType());
		double* pTmp=tmp.GetData<double>();
		double* pX1=x1.GetData<double>();
		double* pX2=x2.GetData<double>();
		for(int i=0;i<x1.Row*x1.Col;i++)
			pTmp[i]=pX1[i]*pX2[i];
		return tmp;
	}

	int save(const char*path,CpkMat&src)
	{
		FILE* fp=fopen(path,"wb");
		if(fp==NULL)
			return PK_NOT_ALLOW_OPERATOR;
		fwrite(&src.Row,1,sizeof(int),fp);
		fwrite(&src.Col,1,sizeof(int),fp);
		fwrite(&src.Depth,1,sizeof(short),fp);
		CpkMat::DATA_TYPE type=src.GetType();
		fwrite(&type,1,sizeof(CpkMat::DATA_TYPE),fp);
		switch(src.GetType())
		{
		case CpkMat::DATA_DOUBLE:
			fwrite(src.GetData<double>(),1,sizeof(double)*src.Row*src.Col*src.Depth,fp);
			break;
		case CpkMat::DATA_INT:
			fwrite(src.GetData<int>(),1,sizeof(int)*src.Row*src.Col*src.Depth,fp);
			break;
		case CpkMat::DATA_BYTE:
			fwrite(src.GetData<BYTE>(),1,sizeof(BYTE)*src.Row*src.Col*src.Depth,fp);
			break;
		}
		fclose(fp);
		return PK_SUCCESS;
	}


	int load(const char*path,CpkMat&dest)
	{
		FILE* fp=fopen(path,"rb+");
		if(fp==NULL)
			return PK_NOT_ALLOW_OPERATOR;
		int row,col;
		short depth;
		CpkMat::DATA_TYPE type;
		fread(&row,1,sizeof(int),fp);
		fread(&col,1,sizeof(int),fp);
		fread(&depth,1,sizeof(short),fp);
		fread(&type,1,sizeof(CpkMat::DATA_TYPE),fp);

		switch(type)
		{
		case CpkMat::DATA_DOUBLE:
			{
				double* pdata=new double[row*col*depth+1];
				fread(pdata,1,sizeof(double)*row*col*depth,fp);
				dest.Resize(row,col,depth,type,pdata);
				delete [] pdata;
			}
			break;
		case CpkMat::DATA_INT:
			{
				int* pdata=new int[row*col*depth+1];
				fread(pdata,1,sizeof(int)*row*col*depth,fp);
				dest.Resize(row,col,depth,type,pdata);
				delete [] pdata;
			}
			break;
		case CpkMat::DATA_BYTE:
			{
				BYTE* pdata=new BYTE[row*col*depth+1];
				fread(pdata,1,sizeof(BYTE)*row*col*depth,fp);
				dest.Resize(row,col,depth,type,pdata);
				delete [] pdata;
			}
			break;
		}
		fclose(fp);
		return PK_SUCCESS;
	}


};

