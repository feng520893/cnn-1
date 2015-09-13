#include "CNNConfig.h"

#include"../layers/GPU/ConvLayer.cuh"
#include"../layers/GPU/PoolLayer.cuh"
#include"../layers/GPU/FullLayer.cuh"
#include"../layers/GPU/SoftMaxLayer.cuh"

#include"../layers/CPU/ConvLayerCPU.h"
#include"../layers/CPU/PoolLayerCPU.h"
#include"../layers/CPU/FullLayerCPU.h"
#include"../layers/CPU/SoftMaxLayerCPU.h"

CCNNConfig::CCNNConfig(void)
{
	inputChannel=1;
	imgDim=inputDim=28;
	bFirstFullLayer=true;
	runMode=PK_CNN_CPU_RUN;
	activeFunType=NL_RELU;
	maxSaveCount=40;
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
			if(mess.find("IMAGE_CHNANEL")!=std::string::npos)
				inputChannel=readInt(mess);
			else if(mess.find("IMAGE_DIM")!=std::string::npos)
				imgDim=inputDim=readInt(mess);
			else if(mess.find("RUN_MODE")!=std::string::npos)
			{
				runMode=readInt(mess);
				if(runMode==PK_CNN_AUTO_RUN||runMode>PK_CNN_GPU_CUDA_RUN)
					runMode=autoRun();
			}
			else if(mess.find("ACTIVEFUN_TYPE")!=std::string::npos)
			{
				activeFunType=readInt(mess);
				if(activeFunType>NL_RELU)
				{
					printf("错误的激活函数设置:%s 强制使用Relu激活函数",mess.c_str());
					activeFunType=NL_RELU;
				}
			}
			else if(mess.find("DATASET_PATH")!=std::string::npos)
			{
				dataSetsPath=readString(mess);
			}
			else if(mess.find("MAX_SAVE_COUNT")!=std::string::npos)
			{
				maxSaveCount=readInt(mess);
			}
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
	do
	{
		getLine(in,mess);
		int index=mess.find("[");
		if(index!=std::string::npos)
			analyticalCNN(in);
	}while(mess.find("}")==std::string::npos&&!in.eof());

	return PK_SUCCESS;
}

int CCNNConfig::analyticalCNN(std::ifstream& in)
{
	std::string mess;
	int type=0;
	CLayer* pBase=NULL;
	do
	{
		getLine(in,mess);
		if(mess.find("LayerType")!=std::string::npos)
		{
			if(mess.find("CONV")!=std::string::npos)
			{
				type=1;
				if(runMode==PK_CNN_CPU_RUN)
					pBase=new CConvLayerCPU();
				else
					pBase=new CConvLayerGPU();
			}
			else if(mess.find("FULL")!=std::string::npos)
			{
				type=2;
				if(runMode==PK_CNN_CPU_RUN)
					pBase=new CFullLayerCPU();
				else
					pBase=new CFullLayerGPU();
			}
			else if(mess.find("SOFTMAX")!=std::string::npos)
			{
				type=3;
				if(runMode==PK_CNN_CPU_RUN)
					pBase=new CSoftmaxLayerCPU();
				else
					pBase=new CSoftMaxLayerGPU();
			}
			else if(mess.find("POOL")!=std::string::npos)
			{
				type=4;
				if(runMode==PK_CNN_CPU_RUN)
					pBase=new CPoolLayerCPU();
				else
					pBase=new CPoolLayerGPU();
			}
			else
			{
				printf("无法识别的信息！ %s",mess.c_str());
			}
		}

		else if(mess.find("NUM_FEATURE")!=std::string::npos)
		{
			pBase->m_curNumFeature=readInt(mess);
			if(pBase->m_curNumFeature==0)
			{
				if(type==3)
				{
					std::vector<std::string> directs;
					pk::findDirectsOrFiles(dataSetsPath.c_str(),directs,true);
					DL_ASSER(!directs.empty());
					pBase->m_curNumFeature=directs.size();
				}
				else
					printf("错误的设置:%s",mess.c_str());
			}
		}
		else if(mess.find("WEIGHT_DECAY")!=std::string::npos)
			pBase->m_lambda=readDouble(mess);

		else if(mess.find("INPUT_NAME")!=std::string::npos)
			pBase->m_inputName=readString(mess);
		else if(mess.find("NAME")!=std::string::npos)
			pBase->m_name=readString(mess);

		else if(mess.find("KERNEL_SIZE")!=std::string::npos)
		{
			if(type==1)
			{
				CConvLayerBase* pTmp=dynamic_cast<CConvLayerBase*>(pBase);
				pTmp->m_kernelDim=readInt(mess);
			}
			else if(type==4)
			{
				CPoolLayerBase* pTmp=dynamic_cast<CPoolLayerBase*>(pBase);
				pTmp->m_kernelDim=readInt(mess);
			}
			else
			{
				printf("错误的设置:%s",mess.c_str());
			}
		}
		else if(mess.find("POOL_TYPE")!=std::string::npos)
		{
			if(type==4)
			{
				CPoolLayerBase* pTmp=dynamic_cast<CPoolLayerBase*>(pBase);
				std::string typeStr=readString(mess);
				if(typeStr.find("MAX")!=std::string::npos)
					pTmp->m_poolType=MAX_POOL;
				else
					pTmp->m_poolType=AVG_POOL;
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
				CFullLayerBase* pTmp=dynamic_cast<CFullLayerBase*>(pBase);
				pTmp->m_dropRate=readDouble(mess);
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

	if(type==1)
	{
		CLayer* pTmp=dynamic_cast<CLayer*>(pBase);
		if(cnnLayers.empty())
			pTmp->m_inputNumFeature=inputChannel;
		else
			pTmp->m_inputNumFeature=cnnLayers[cnnLayers.size()-1]->m_curNumFeature;
		pTmp->batch=sgd.minibatch;

		CConvLayerBase* pTmp2=dynamic_cast<CConvLayerBase*>(pBase);
		inputDim=inputDim-pTmp2->m_kernelDim+1;
		pTmp2->m_convDim=inputDim;
	}
	else if(type==4)
	{
		CLayer* pTmp=dynamic_cast<CLayer*>(pBase);
		CPoolLayerBase* pTmp2=dynamic_cast<CPoolLayerBase*>(pBase);
		for(int j=0;j<cnnLayers.size();j++)
		{
			if(pTmp->m_inputName==cnnLayers[j]->m_name)
			{
				CConvLayerBase* pTmpC=dynamic_cast<CConvLayerBase*>(cnnLayers[j]);
				CLayer* pTmpC2=dynamic_cast<CLayer*>(cnnLayers[j]);
				pTmp->batch=sgd.minibatch;
				pTmp2->m_preConvDim=pTmpC->m_convDim;
				pTmp->m_curNumFeature=pTmpC2->m_curNumFeature;
				assert(dataDim%poolLayers[j].m_poolArea==0);
				inputDim/=pTmp2->m_kernelDim;
				break;
			}
		}
	}
	else if(type==2||type==3)
	{
		CLayer* pTmp=dynamic_cast<CLayer*>(pBase);
		if(bFirstFullLayer)
		{
			pTmp->m_inputNumFeature=inputDim*inputDim*cnnLayers[cnnLayers.size()-1]->m_curNumFeature;
			bFirstFullLayer=false;
		}
		else
			pTmp->m_inputNumFeature=cnnLayers[cnnLayers.size()-1]->m_curNumFeature;
		pTmp->batch=sgd.minibatch;
	}
	else
	{
	}
	pBase->initMem();
	cnnLayers.push_back(pBase);
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

bool CCNNConfig::isCUDA()
{
    int count;
    cudaGetDeviceCount(&count);
    if(count == 0) 
	{
        fprintf(stderr, "找不到N卡\n");
        return false;
    }
    int i;
    for(i = 0; i < count; i++) 
	{
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) 
		{
            if(prop.major >= 1) 
                break;
        }
    }
    if(i == count) 
	{
        fprintf(stderr, "当前驱动程序不支持CUDA\n");
        return false;
    }
    cudaSetDevice(i);
    return true;
}

int CCNNConfig::autoRun()
{
	if(isCUDA())
		return PK_CNN_GPU_CUDA_RUN;
	return PK_CNN_CPU_RUN;
}