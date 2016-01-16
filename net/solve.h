#ifndef SOLVE_H
#define SOLVE_H
#include"net.h"
#include"ReadConfig\ReadConfig.h"
class CSolve
{
public:
	CSolve();
	~CSolve();
	int init(const char* configPath);
	int run();
private:
	int train();
	int test();
//	int predict(std::vector<CpkMat>& imgs,std::vector<double>&pred,int topNum=5);
	precision getLearningRate();
	CNET m_trainNet,m_testNet;
	solveParams m_params;
	int m_currentStep,m_currentIter;
};
#endif