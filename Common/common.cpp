#include"common.h"
#include<stdlib.h>
#include<time.h>
int randn(double* data,int dataSize,float rate)
{
	srand(time(NULL));
	for(int i=0;i<dataSize;i++)
	{
		double V1, V2, S;
		int phase = 0;
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
		data[i]=X*rate;
	}
	return 0;
}

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