#pragma once
#include<fstream>
#include <string>
#include<sstream>
#include "../common/pkDefine.h"
#include"../layers/GPU/LayerBaseGPU.cuh"

class CCNNConfig
{
public:
	CCNNConfig(void);
	virtual ~CCNNConfig(void);
	int loadConfig(const char* path);
	int saveConfig(const char* path=NULL);
	int  writeString(const char* key,std::string mess);
private:
	bool isCUDA();
	int autoRun();
	int loadGlobalMess(std::ifstream& in);
	int loadMinSGDMess(std::ifstream& in);
	int loadCnnMess(std::ifstream& in);
	bool readBool(std::string& mess);
	int  readInt(std::string& mess);
	std::string  readString(std::string& mess);
	
	double readDouble(std::string&mess);

	std::string& trim(std::string &s);

	int analyticalCNN(std::ifstream& in);

	std::string& getLine(std::ifstream& in,std::string& mess);

	std::string        filePath;
	std::string        fileText;

	bool               bFirstFullLayer;
    int                inputChannel;
	int                inputDim;
public:
	MinSGD sgd;
	std::vector<CLayer*>    cnnLayers;
	int                     runMode;
	int                     activeFunType;
	std::string             dataSetsPath;
	int                     imgDim;
	int                     maxSaveCount;
};
