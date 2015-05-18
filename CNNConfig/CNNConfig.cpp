#include "CNNConfig.h"

CCNNConfig::CCNNConfig(void)
{
	gradCheck=false;
	inputChannel=1;
	inputDim=28;
}

CCNNConfig::~CCNNConfig(void)
{
}

int CCNNConfig::loadConfig(const char* path)
{
	std::ifstream in(path);
	if(!in.is_open())
		return PK_NOT_FILE;
	std::ostringstream  oss;
	oss<<in.rdbuf();
	fileText=oss.str();

	filePath=path;

	in.seekg(0,std::ios::beg);

	int nRet=PK_SUCCESS;
	do
	{
		std::string mess;
		getLine(in,mess);
		if(mess.find("GLOBAL{")!=std::string::npos)
			nRet=loadGlobalMess(in);
		else if(mess.find("MINI_SGD{")!=std::string::npos)
			nRet=loadMinSGDMess(in);
		else if(mess.find("CNN{")!=std::string::npos)
			nRet=loadCnnMess(in);
		else
		{
		}
	}while(!in.eof()&&nRet==PK_SUCCESS);
	in.close();
	return nRet;
}

int CCNNConfig::saveConfig(const char* path)
{
	std::string savePath;
	if(path==NULL)
		savePath=filePath;
	else
		savePath=path;
	std::ofstream out(savePath);
	out<< fileText;
	out.close();
	return PK_SUCCESS;
}

int CCNNConfig::loadGlobalMess(std::ifstream& in)
{
	std::string mess;
	do
	{
		getLine(in,mess);
		int index=mess.find("=");
		if(index!=std::string::npos)
		{
			if(mess.find("GRAD_CHECK")!=std::string::npos)
				gradCheck=readBool(mess);
			else if(mess.find("IMAGE_CHNANEL")!=std::string::npos)
				inputChannel=readInt(mess);
			else if(mess.find("IMAGE_DIM")!=std::string::npos)
				inputDim=readInt(mess);
			else
			{
				printf("无法解析的数据:%s",mess.c_str());
			}
		}
	}while(mess.find("}")==std::string::npos&&!in.eof());
	return PK_SUCCESS;
}

int CCNNConfig::loadMinSGDMess(std::ifstream& in)
{
	std::string mess;
	do
	{
		getLine(in,mess);
		int index=mess.find("=");
		if(index!=std::string::npos)
		{
			if(mess.find("ALPHA")!=std::string::npos)
				sgd.alpha=readDouble(mess);
			else if(mess.find("MOMENTUM")!=std::string::npos)
				sgd.momentum=readDouble(mess);
			else if(mess.find("MINI_BATCH")!=std::string::npos)
				sgd.minibatch=readInt(mess);
			else if(mess.find("EPOCHES")!=std::string::npos)
				sgd.epoches=readInt(mess);
			else
			{
				printf("无法解析的数据:%s",mess.c_str());
			}
		}
	}while(mess.find("}")==std::string::npos&&!in.eof());
	return PK_SUCCESS;
}

int CCNNConfig::loadCnnMess(std::ifstream& in)
{
	std::string mess;
	int type=0;
	CLayer* baseLayer=NULL;
	do
	{
		getLine(in,mess);
		int index=mess.find("[");
		if(index!=std::string::npos)
			type=analyticalCNN(&baseLayer,in);
		if(type==1)
		{
			CConvLayer* pTmp=(CConvLayer*)baseLayer;
			convs.push_back(*pTmp);
			delete baseLayer;
			baseLayer=NULL;
			type=0;
		}
		else if(type==2)
		{
			CFullLayer* pTmp=(CFullLayer*)baseLayer;
			fulls.push_back(*pTmp);
			delete baseLayer;
			baseLayer=NULL;
			type=0;
		}
		else if(type==3)
		{
			CSoftMaxLayer* pTmp=(CSoftMaxLayer*)baseLayer;
			sfm=*pTmp;
			delete baseLayer;
			baseLayer=NULL;
			type=0;
		}
		else
		{
		}
	}while(mess.find("}")==std::string::npos&&!in.eof());
	
	return PK_SUCCESS;
}

int CCNNConfig::analyticalCNN(CLayer** ppBase,std::ifstream& in)
{
	std::string mess;
	int type=0;
	do
	{
		getLine(in,mess);
		if(mess.find("LayerType")!=std::string::npos)
		{
			if(mess.find("CONV")!=std::string::npos)
			{
				type=1;
				*ppBase=new CConvLayer();
			}
			else if(mess.find("FULL")!=std::string::npos)
			{
				type=2;
				*ppBase=new CFullLayer();
			}
			else if(mess.find("SOFTMAX")!=std::string::npos)
			{
				type=3;
				*ppBase=new CSoftMaxLayer();
			}
			else
			{
				printf("无法识别的信息！ %s",mess.c_str());
			}
		}

		else if(mess.find("NUM_FEATURE")!=std::string::npos)
			(*ppBase)->m_curNumFeature=readInt(mess);
		else if(mess.find("WEIGHT_DECAY")!=std::string::npos)
			(*ppBase)->m_lambda=readDouble(mess);

		else if(mess.find("KERNEL_SIZE")!=std::string::npos)
		{
			if(type==1)
			{
				CConvLayer* pTmp=(CConvLayer*)(*ppBase);
				pTmp->m_maskDim=readInt(mess);
			}
			else
			{
				printf("错误的设置:%s",mess.c_str());
			}
		}
		else if(mess.find("DROPCONNECT_RATE")!=std::string::npos)
		{
			if(type==2)
			{
				CFullLayer* pTmp=(CFullLayer*)(*ppBase);
				pTmp->m_rate=readDouble(mess);
			}
			else
			{
				printf("错误的设置:%s",mess.c_str());
			}
		}
		else
		{
			///未定义数据不处理
		}
	}while(mess.find("]")==std::string::npos&&!in.eof());
	return type;
}

bool CCNNConfig::readBool(std::string& mess)
{
	if(mess.find("false")!=std::string::npos)
		return false;
	return true;
}

int CCNNConfig::readInt(std::string& mess)
{
	int index=mess.find("=")+1;
	mess=mess.substr(index,mess.length()-index);
	trim(mess);
	return atoi(mess.c_str());
}

double CCNNConfig::readDouble(std::string& mess)
{
	int index=mess.find("=")+1;
	mess=mess.substr(index,mess.length()-index);
	trim(mess);
	return atof(mess.c_str());
}

std::string CCNNConfig::readString(std::string& mess)
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

int CCNNConfig::writeString(const char* key,std::string mess)
{
	int index=fileText.find(key);
	index=fileText.find('=',index)+1;
	int index2=fileText.find('}',index);
	mess+=" \n\r";
	fileText.replace(index,index2-index,mess);
	
	return PK_SUCCESS;
}

std::string& CCNNConfig::trim(std::string &s) 
{
	if (s.empty()) 
		return s;
	s.erase(0,s.find_first_not_of(" "));
	s.erase(s.find_last_not_of(" ") + 1);
	return s;
}

std::string& CCNNConfig::getLine(std::ifstream& in,std::string& mess)
{
	do{
		getline(in,mess);
	}while(mess.empty()&&!in.eof());
	return mess;
}