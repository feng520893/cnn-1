#pragma once
#include"..\layers\\ConvLayer.cuh"
#include"..\layers\\FullLayer.cuh"
#include"..\layers\\SoftMaxLayer.cuh"

#include"..\common\\cpkMat.h"

#include"../CNNConfig/CNNConfig.h"

#include<Windows.h>

typedef	CpkMat Data;

class CCNN
{
public:
	CCNN(void);
	virtual ~CCNN(void);
	void setParam(MinSGD& sgd,float dropRate,int activeType);
	const MinSGD& getMinSGDInfo();
	int init(std::vector<CConvLayer>& convLayers,std::vector<CFullLayer>&fullLayers,CSoftMaxLayer& sfm,int dataDim,short dataChannel,int batch);
	int init(const char*path);
	int cnnTrain(Data&datas,std::vector<int>& labels);
	int cnnRun(std::vector<double>&pred,Data&datas);
	double computeNumericalGradient(Data&datas,std::vector<int>& labels);

	int save(const char* path);
	int load(const char* path);
	
private:
	MinSGD m_sgd;
	int    m_activeType;
	float  m_dropRate;
	std::vector<CConvLayer> m_convLayers;
	std::vector<CFullLayer> m_fullLayers;
	CSoftMaxLayer m_sfm;
	int feedforward(std::vector<double>&predMat,double*data,bool bPred);
	double getCost(std::vector<int>& labels);
	int backpropagation(std::vector<int>& labels);
	double cnnCost(std::vector<double>&predMat,double*data,std::vector<int>& labels,bool bPred=false);
	int grads(double* srcData);
	int updateWeights();
};
