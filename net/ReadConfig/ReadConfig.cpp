#include "ReadConfig.h"
#include<algorithm>

CReadConfig::CReadConfig(void)
{
	solveParams.runMode=PK_CNN_CPU_RUN;
	solveParams.testInterval=solveParams.saveInterval=40;
	solveParams.LR_Policy="FIXED";
}

CReadConfig::~CReadConfig(void)
{
}

int CReadConfig::loadConfig(const char* path)
{
	std::ifstream in(path);
	if(!in.is_open())
		return NET_FILE_NOT_EXIST;

	int nRet=NET_SUCCESS;
	do
	{
		std::string mess;
		getLine(in,mess);
		if(mess.find("GLOBAL{")!=std::string::npos)
			nRet=loadGlobalMess(in);
		else if(mess.find("CNN{")!=std::string::npos)
			nRet=loadNetMess(in);
		else
		{
		}
	}while(!in.eof()&&nRet==PK_SUCCESS);
	in.close();
	return nRet;
}

int CReadConfig::loadGlobalMess(std::ifstream& in)
{
	std::string mess;
	do
	{
		getLine(in,mess);
		int index=mess.find("=");
		if(index!=std::string::npos)
		{
			if(mess.find("RUN_MODE")!=std::string::npos)
			{
				solveParams.runMode=readInt(mess);
			}
			else if(mess.find("NET_MODE")!=std::string::npos)
			{
				solveParams.netMode=readInt(mess);
			}
			
			else if(mess.find("SAVE_INTERVAL")!=std::string::npos)
			{
				solveParams.saveInterval=readInt(mess);
			}
			else if(mess.find("MODEL_SAVE_PATH")!=std::string::npos)
			{
				solveParams.saveModelPath=readString(mess);
			}
			else if(mess.find("TEST_INTERVAL")!=std::string::npos)
			{
				solveParams.testInterval=readInt(mess);
			}
			else if(mess.find("BASE_LR")!=std::string::npos)
				solveParams.baseLR=readDouble(mess);
			else if(mess.find("LR_POLICY")!=std::string::npos)
				solveParams.LR_Policy=readString(mess);
			else if(mess.find("GAMMA")!=std::string::npos)
				solveParams.gamma=readDouble(mess);
			else if(mess.find("POWER")!=std::string::npos)
				solveParams.power=readDouble(mess);
			else if(mess.find("STEP_SIZE")!=std::string::npos)
				solveParams.stepSize=readInt(mess);
			else if(mess.find("MAX_ITER")!=std::string::npos)
				solveParams.maxIter=readInt(mess);
			else if(mess.find("TEST_ITER")!=std::string::npos)
				solveParams.testIter=readInt(mess);
			else if(mess.find("MOMENTUM")!=std::string::npos)
				solveParams.momentum=readDouble(mess);
			else if(mess.find("DISPLAY")!=std::string::npos)
				solveParams.display=readInt(mess);
			else
			{
				printf("无法解析的数据:%s",mess.c_str());
			}
		}
	}while(mess.find("}")==std::string::npos&&!in.eof());

	return NET_SUCCESS;
}

int CReadConfig::loadNetMess(std::ifstream& in)
{
	int nRet=0;
	std::string mess;
	do
	{
		getLine(in,mess);
		int index=mess.find("[");
		if(index!=std::string::npos)
			nRet=analyticalNet(in);
	}while(nRet==0&&mess.find("}")==std::string::npos&&!in.eof());

	return nRet;
}

int CReadConfig::analyticalNet(std::ifstream& in)
{
	std::string mess;
	int nRet=0;
	LayerParam param;

	do
	{
		getLine(in,mess);
		if(mess.find("LAYER_TYPE")!=std::string::npos)
			param.typeName=readString(mess);

		else if(mess.find("NUM_FEATURE")!=std::string::npos)
		{
			param.curNumFeature=readInt(mess);
		}
		else if(mess.find("WEIGHT_DECAY")!=std::string::npos)
			param.lambda=readDouble(mess);

		else if(mess.find("INPUT_NAME")!=std::string::npos)
			param.inputName.push_back(readString(mess));
		else if(mess.find("OUTPUT_NAME")!=std::string::npos)
			param.outputName.push_back(readString(mess));
		else if(mess.find("NAME")!=std::string::npos)
			param.name=readString(mess);

		else if(mess.find("KERNEL_SIZE")!=std::string::npos)
		{
			if(param.typeName=="CONV")
				param.convParam.kernelDim=readInt(mess);
			else if(param.typeName=="POOL")
				param.poolParam.kernelDim=readInt(mess);
			else
			{
				printf("错误的设置:%s",mess.c_str());
			}
		}
		else if(mess.find("POOL_TYPE")!=std::string::npos)
		{
			std::string typeStr=readString(mess);
			if(typeStr.find("MAX")!=std::string::npos)
				param.poolParam.poolType=MAX_POOL;
			else
				param.poolParam.poolType=AVG_POOL;
		}
		else if(mess.find("DROPOUT_RATE")!=std::string::npos)
		{
			param.dropoutParam.dropoutRate=readDouble(mess);
		}
		else if(mess.find("NORM_REGION_TYPE")!=std::string::npos)
		{
			std::string tmp=readString(mess);
			if(tmp == "ACROSS_CHANNELS")
				param.lrnParam.normRegionType=ACROSS_CHANNELS;
		}
		else if(mess.find("LOCAL_SIZE")!=std::string::npos)
		{
			param.lrnParam.localSize=readInt(mess);
		}
		else if(mess.find("ALPHA")!=std::string::npos)
		{
			param.lrnParam.alpha=readDouble(mess);
		}
		else if(mess.find("BEAT")!=std::string::npos)
		{
			param.lrnParam.beat=readDouble(mess);
		}
		else if(mess.find("PHASE")!=std::string::npos)
		{
			std::string name=readString(mess);
			if(name=="TRAIN")
				param.phase=TRAIN;
			else
				param.phase=TEST;
		}
		else if(mess.find("DATASET_PATH")!=std::string::npos)
		{
			param.dataParam.dataPath=readString(mess);
		}
		else if(mess.find("MEAN_DATA_PATH")!=std::string::npos)
		{
			param.dataParam.meanDataPath=readString(mess);
		}
		else if(mess.find("MEAN_DATA_MODE")!=std::string::npos)
		{
			param.dataParam.geneMeanImgMode=readInt(mess);
		}
		else if(mess.find("IS_GRAY")!=std::string::npos)
		{
			param.dataParam.bGray=readBool(mess);
		}
		else if(mess.find("IS_MOVE_DATA")!=std::string::npos)
		{
			param.dataParam.bMove=readBool(mess);
		}
		else if(mess.find("DATA_DIM")!=std::string::npos)
		{
			param.dataParam.dataDim=readInt(mess);
		}
		else if(mess.find("BATCH_SIZE")!=std::string::npos)
		{
			param.dataParam.batchSize=readInt(mess);
		}
		else if(mess.find("CREATE_MEAN_IMG_MODE")!=std::string::npos)
		{
			param.dataParam.geneMeanImgMode=readInt(mess);
		}
		else if(mess.find("DATASET_TYPE")!=std::string::npos)
		{
			std::string name=readString(mess);
			if(name=="MNIST")
				param.dataParam.datasetType=1;
			else if(name=="CIFAR")
				param.dataParam.datasetType=2;
			else
				param.dataParam.datasetType=3;
		}
		else
		{
			///未定义数据不处理
		}
	}while(mess.find("]")==std::string::npos&&!in.eof());
	
	layerParams.push_back(param);

	return nRet;
}

bool CReadConfig::readBool(std::string& mess)
{
	if(mess.find("0")!=std::string::npos)
		return false;
	if(mess.find("FALSE")!=std::string::npos)
		return false;
	return true;
}

int CReadConfig::readInt(std::string& mess)
{
	int index=mess.find("=")+1;
	mess=mess.substr(index,mess.length()-index);
	trim(mess);
	return atoi(mess.c_str());
}

double CReadConfig::readDouble(std::string& mess)
{
	int index=mess.find("=")+1;
	mess=mess.substr(index,mess.length()-index);
	trim(mess);
	return atof(mess.c_str());
}

std::string CReadConfig::readString(std::string& mess)
{
	int index=mess.find("=")+1;
	int index2=mess.find('}');
	if(index2!=std::string::npos)
		mess=mess.substr(index,index2-index);
	else
		mess=mess.substr(index,mess.length()-index);
	trim(mess);
	
	std::string tmp=mess;
	return tmp;
}

std::string& CReadConfig::trim(std::string &s) 
{
	if (s.empty()) 
		return s;
	s.erase(0,s.find_first_not_of(" "));
	s.erase(s.find_last_not_of(" ") + 1);
	return s;
}

std::string& CReadConfig::getLine(std::ifstream& in,std::string& mess)
{
	do{
		getline(in,mess);
		trim(mess);
		std::transform(mess.begin(),mess.end(),mess.begin(),toupper);
	}while((mess.find("//")!=std::string::npos||mess.empty())&&!in.eof());
	return mess;
}