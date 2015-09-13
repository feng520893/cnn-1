#ifndef LAYER_BASE_CPU_H
#define LAYER_BASE_CPU_H
#include "../LayerBase.h"
#include "ProcessPool.h"

#ifndef CPU_FREE
#define CPU_FREE(x)if(x)delete [] x
#endif

struct CLayerBaseCPU : public CLayer
{
	CLayerBaseCPU(){};
	~CLayerBaseCPU(){};

	virtual void   setWeightValue(int index,double value);
	virtual double getWeightValue(int index);

	virtual void setBiasValue(int index,double value);
	virtual double getBiasValue(int inddex);

	virtual void getWeightsGrad(double* gradWeightData);

	virtual void getBiasGrad(double* gradBiasData);

	void updateWeight(float mom,float alpha);

	virtual int    save(FILE* fp);
	virtual int    load(FILE*fp);

	virtual int    initMem();
	virtual void   freeMem();

	int matrixMul(precision* x,int rowL,int colL,precision* y,int rowR,int colR,precision* z);

	double getWeightCost();

};

double activeFunCPU(double src,int type);

double d_activeFunCPU(double src,int type);

struct nolineData
{
	double* deltaE;
	double* activeData;
	double* deltaS;
	int type;
};

unsigned int __stdcall NolineProcessThread(LPVOID pM);

unsigned int __stdcall GetWeightsCostThread(LPVOID pM);

#endif
