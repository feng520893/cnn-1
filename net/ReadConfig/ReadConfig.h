#pragma once
#include<fstream>
#include"../net.h"
#include"../../Common/PKDefine.h"

struct solveParams
{
	solveParams()
	{
		stepSize=saveInterval=testInterval=runMode=netMode=0;
	    momentum=0.9;
	    baseLR=0.01;
	    gamma=0.0001;
	    power=0.75;
	    batchSize=200;
		maxIter=30000;
	    testIter=50;
	    display=300;
	};
	int netMode;
	int runMode;
	int saveInterval;
	int testInterval;
	float momentum;
	float baseLR;
	std::string LR_Policy;
	float gamma;
	float power;
	unsigned int stepSize;
	int   batchSize;
	unsigned int maxIter;
	unsigned int testIter;
	unsigned int display;
	std::string saveModelPath;
};

class CReadConfig
{
public:
	CReadConfig(void);
	virtual ~CReadConfig(void);
	int loadConfig(const char* path);

private:
	int loadGlobalMess(std::ifstream& in);
	int loadNetMess(std::ifstream& in);
	bool readBool(std::string& mess);
	int  readInt(std::string& mess);
	std::string  readString(std::string& mess);
	
	double readDouble(std::string&mess);

	std::string& trim(std::string &s);

	int analyticalNet(std::ifstream& in);

	std::string& getLine(std::ifstream& in,std::string& mess);
public:
	solveParams                solveParams;
	std::vector<LayerParam>    layerParams;
};
