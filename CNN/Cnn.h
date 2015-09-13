#pragma once

#include"..\common\\cpkMat.h"
#include"..\\layers\\LayerBase.h"

#include<vector>

typedef	CpkMat Data;

class CCNN
{
public:
	CCNN(void);
	virtual ~CCNN(void);
	void setParam(MinSGD& sgd,int activeType,int rumMode);
	const MinSGD& getMinSGDInfo();
	int init(std::vector<CLayer*>& cnnLayers);
	int cnnTrain(Data&datas,std::vector<int>& labels,const char* savePath=NULL,int maxSaveCount=0);
	int cnnRun(std::vector<double>&pred,Data&datas);
	double computeNumericalGradient(Data&datas,std::vector<int>& labels);

	int save(const char* path);
	int load(const char* path);
	
private:
	MinSGD m_sgd;
	int    m_activeType,m_runMode;

	std::vector<CLayer*> m_cnn;   //单向组合网络
	int feedforward(std::vector<double>&predMat,double*data,bool bPred);
	double getCost(std::vector<int>& labels);
	int backpropagation(std::vector<int>& labels);
	double cnnCost(std::vector<double>&predMat,double*data,std::vector<int>& labels,bool bPred=false);
	int grads(double* srcData);
	int updateWeights();
};
