#ifndef CONVLAYER_H
#define CONVLAYER_H
#include"LayerBase.cuh"

class CConvLayer : public CLayer
{
public:
	CConvLayer()
	{
		m_poolData=m_convData=m_convNoActiveData=m_poolDelta=m_weightTmp=NULL;
		m_maxIndexData=NULL;
		m_poolArea=2;
	}
	~CConvLayer(){freeMem();};
	void    feedforward(double* srcData,int activeType);
	int     backpropagation(double*preDelta,int activeType);
	void    getGrad(double* srcData);
 	void    updateWeight(float mom,float alpha);
 	double  getCost();
	int     initMem();
	void    freeMem();
	

	void getWeightsGrad(double* CPUdata)
	{
		cudaMemcpy(CPUdata,m_weightGrad,sizeof(double)*m_inputNumFeature*m_curNumFeature*m_maskDim*m_maskDim,cudaMemcpyDeviceToHost);
	}

	void getBiasGrad(double* CPUdata)
	{
		cudaMemcpy(CPUdata,m_biasGrad,sizeof(double)*m_curNumFeature,cudaMemcpyDeviceToHost);
	}

	int     m_convDim;
	short   m_maskDim;
	short   m_poolArea;
	double* m_poolData;
	double* m_poolDelta;

private:
	double* m_convData;
	double* m_convNoActiveData;
	double* m_weightTmp;
	int*    m_maxIndexData;
};

#endif