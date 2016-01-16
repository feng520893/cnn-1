#ifndef COMMON_H
#define COMMON_H

#include<time.h>

#define NL_SIGMOID   0
#define NL_TANH      1 
#define NL_RELU      2

#define MAX_POOL 0
#define AVG_POOL 1
#define RAND_POOL 2

#define RUN_CPU 0
#define RUN_GPU 1

#define FILE_MAX_PATH   512

#define PK_SUCCESS 0

#ifndef PK_MAX
#define PK_MAX(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef PK_MIN
#define PK_MIN(a,b)            (((a) < (b)) ? (a) : (b))
#endif

template<typename _Tp> 
struct _tDLparam
{
	_tDLparam():pred(false),predData(NULL){};
	_tDLparam(std::vector<_Tp> labels):predData(NULL)
	{
		this->labels=labels;
	};

	~_tDLparam()
	{
		if(predData!=NULL)
			delete [] predData;
	}
	std::vector<_Tp> labels;
	bool pred;
	double* predData;
};

typedef _tDLparam<int> DLparam;

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

	int findDirectsOrFiles(std::string direct,std::vector<std::string>& files,const char* extension=NULL,bool bOnlyFindDirect=false);
};

#endif