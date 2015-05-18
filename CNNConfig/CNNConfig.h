#pragma once
#include<fstream>
#include <string>
#include<sstream>
#include "../common/pkDefine.h"
#include"../layers/ConvLayer.cuh"
#include"../layers/FullLayer.cuh"
#include"../layers/SoftMaxLayer.cuh"
class CCNNConfig
{
public:
	CCNNConfig(void);
	virtual ~CCNNConfig(void);
	int loadConfig(const char* path);
	int saveConfig(const char* path=NULL);
	int  writeString(const char* key,std::string mess);
private:
	int loadGlobalMess(std::ifstream& in);
	int loadMinSGDMess(std::ifstream& in);
	int loadCnnMess(std::ifstream& in);
	bool readBool(std::string& mess);
	int  readInt(std::string& mess);
	std::string  readString(std::string& mess);
	
	double readDouble(std::string&mess);

	std::string& trim(std::string &s);

	int analyticalCNN(CLayer** ppBase,std::ifstream& in);

	std::string& getLine(std::ifstream& in,std::string& mess);

	std::string        filePath;
	std::string        fileText;
public:
	MinSGD sgd;
	std::vector<CConvLayer> convs;
	std::vector<CFullLayer> fulls;
	CSoftMaxLayer           sfm;
	bool                    gradCheck;
	int                     inputChannel;
	int                     inputDim;
};
