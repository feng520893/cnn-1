#ifndef COMMON_H
#define COMMON_H
#include"pkdefine.h"
#include"pkStruct.h"
#include<vector>
#include<string>
#include<time.h>
namespace pk
{
	//高斯随机分布
	template<typename T>
	int randn(T* data,int dataSize,float rate=0.1)
	{
		srand(time(NULL));
		for(int i=0;i<dataSize;i++)
		{
			T V1, V2, S;
			int phase = 0;
			T X;
			if ( phase == 0 ) 
			{
				do {
					T U1 = (T)::rand() / RAND_MAX;
					T U2 = (T)::rand() / RAND_MAX;
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

	//普通随机分布
	template<typename T>
	int rand(T* data,int dataSize)
	{
		srand(time(NULL));
		for(int i=0;i<dataSize;i++)
		{
			data[i]=(T)::rand() / RAND_MAX;
		}
		return 0;
	}

	void transpose(double** data,int& row,int& col);
	double* matrixMul(double* left,int leftRow,int leftCol,double* right,int rightRow,int rightCol);

	int findDirectsOrFiles(std::string direct,std::vector<std::string>& files,bool bOnlyFindDirect=false);
};
#endif