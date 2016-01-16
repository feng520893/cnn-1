#include "net.h"
#include"..\\common\\pkmatFunctions.h"
#include"../Common/imgProc/PKImageFunctions.h"

#include"layerFactory.h"

CNET::CNET(void)
{
}

CNET::~CNET(void)
{
	for(int i=0;i<m_layers.size();i++)
		delete m_layers[i];
	for(int i=0;i<m_blobs.size();i++)
		delete m_blobs[i];
	m_blobs.clear();
}

int CNET::init(std::vector<LayerParam>& layerParams)
{
	int nRet=NET_SUCCESS;
	openNet::CLayerFactory* pFactory=openNet::CLayerFactory::initialize();

	for(int i=0;i<m_blobs.size();i++)
	{
		m_blobs[i]->destroy();
		delete m_blobs[i];
	}
	m_blobs.clear();
	m_blobNameToID.clear();
	m_layerNameToID.clear();

	m_forwardInputVec.resize(layerParams.size());
	m_forwardOutputVec.resize(layerParams.size());
	m_layerNeedBack.resize(layerParams.size());

	int count=0;
	for(int i=0;i<layerParams.size();i++)
	{
		CLayer* pLayer=NULL;
		LayerParam layerParam=layerParams[i];
		if(layerParam.phase == EMPTY)
			layerParam.phase=m_phase;

		if(layerParam.name.empty())
		{
			char name[256]={0};
			sprintf(name,"auto%d",++count);
			layerParam.name=name;
		}

		pLayer=pFactory->createLayer(layerParam);
		if(pLayer == NULL)
		{
			nRet=NET_CREATE_LAYER_FAILE;
			break;
		}

		//连接输入层
		nRet=appendInput(layerParam,i);

		//连接输出层
		nRet=appendOutput(layerParam,i);
		if(nRet!=0)
			return nRet;

		//设置反向传递标志
		if(i == 0 ||m_forwardInputVec[i-1].empty())
			m_layerNeedBack[i].push_back(false);
		else
			m_layerNeedBack[i].push_back(true);

		m_layers.push_back(pLayer);
		m_layerNameToID[pLayer->m_param.name]=m_layers.size()-1;
	}

	//初始化网络
	for(int i=0;i<m_layers.size();i++)
	{
		nRet=m_layers[i]->setup(m_forwardInputVec[i],m_forwardOutputVec[i]);
		if(nRet!=NET_SUCCESS)
		{
			printf("第%d层初始化失败,错误码:%d",i,nRet);
			break;
		}
	}
	return nRet;
}

int CNET::shareDataTo(CNET& dest)
{
	for(int i=0;i<m_layers.size();i++)
	{
		if(dest.m_layers[i]->m_param.name == m_layers[i]->m_param.name)
		{
			for(int j=0;j<m_layers[i]->m_layerBlobs.size();j++)
			{
				dest.m_layers[i]->m_layerBlobs[j]->shareDataFrom(m_layers[i]->m_layerBlobs[j]);
			}
		}
	}
	return NET_SUCCESS;
}

double CNET::computeNumericalGradient()
{
	int nRet=NET_SUCCESS;

	step();

	unsigned int size=0;
	for(int i=0;i<m_layers.size();i++)
		size+=m_layers[i]->m_weightLen+m_layers[i]->m_param.curNumFeature;

	double* numgrad=new double[size]; 
	//拷贝梯度差
	double* grad=new double[size];
	unsigned int index=0;
	for(int i=0;i<m_layers.size();i++)
	{
		if(m_layers[i]->m_weightLen==0)
			continue;
		m_layers[i]->getWeightsGrad(grad+index);
		index+=m_layers[i]->m_weightLen;
		m_layers[i]->getBiasGrad(grad+index);
		index+=m_layers[i]->m_param.curNumFeature;
	}

	double epsilon = 1e-4;

	int index2=0;
	for(int i=0;i<m_layers.size();i++)
	{
		int weightSize=m_layers[i]->m_weightLen;

		if(weightSize==0)
			continue;

		int biasSize=m_layers[i]->m_param.curNumFeature;
		for(int j=0;j<weightSize;j++)
		{
			double oldT = m_layers[i]->getWeightValue(j);
			m_layers[i]->setWeightValue(j,oldT+epsilon);
			double pos = step();
			m_layers[i]->setWeightValue(j,oldT-epsilon);
			double neg = step();
			numgrad[index2++]= (pos-neg)/(2*epsilon);
			m_layers[i]->setWeightValue(j,oldT);
		}
		for(int j=0;j<biasSize;j++)
		{
			double oldT = m_layers[i]->getBiasValue(j);
			m_layers[i]->setBiasValue(j,oldT+epsilon);
			double pos = step();
			m_layers[i]->setBiasValue(j,oldT-epsilon);
			double neg = step();
			numgrad[index2++]= (pos-neg)/(2*epsilon);
			m_layers[i]->setBiasValue(j,oldT);
		}
	}

	for(int i=0;i<index;i++)
	{
		if(i<52)
			printf("%.6lf   %.6lf\n",numgrad[i],grad[i]);
		else if(i<256)
			printf("%.6lf   %.6lf\n",numgrad[i],grad[i]);
		else if(i<716)
			printf("%.6lf   %.6lf\n",numgrad[i],grad[i]);
		else
			printf("%.6lf   %.6lf\n",numgrad[i],grad[i]);
	}
	double sum1=0;
	for(int i=0;i<size;i++)
		sum1+=pow(numgrad[i]-grad[i],2);
	sum1=sqrt(sum1);

	double sum2=0;
	for(int i=0;i<size;i++)
		sum2+=::pow(numgrad[i]+grad[i],2);
	sum2=sqrt(sum2);

	delete [] numgrad;
	delete [] grad;

	return sum1/sum2;
}

precision CNET::step()
{
	precision cost;
	feedforward(&cost);
	if(m_phase!=TEST)
		backpropagation();
	return cost;
}

int CNET::feedforward(precision* cost)
{
	precision nRet=0.0;
	for(int i=0;i<m_layers.size();i++)
	{
		nRet=m_layers[i]->feedforward(m_forwardInputVec[i],m_forwardOutputVec[i]);
		if((int)nRet<NET_SUCCESS)
		{
			printf("第%d层的前向传递失败，错误码%d\n",i,(int)nRet);
			break;
		}
		cost[0]+=nRet;
	}
	return nRet;
}

int CNET::backpropagation()
{
	int nRet=NET_SUCCESS;
	for(int i=m_layers.size()-1;i>0;i--)
	{
		nRet=m_layers[i]->backpropagation(m_forwardOutputVec[i],m_layerNeedBack[i],m_forwardInputVec[i]);
		if(nRet!=NET_SUCCESS)
		{
			printf("第%d层的反向传递失败，错误码%d",i,nRet);
			break;
		}
	}

	return nRet;
}

int CNET::updateWeights(float momentum,float lr)
{
	for(int i=0;i<m_layers.size();i++)
		m_layers[i]->updateWeight(momentum,lr);
	return NET_SUCCESS;
}

int CNET::clearDiff()
{
	for(int i=0;i<m_blobs.size();i++)
	{
		cudaError_t c=cudaMemset(m_blobs[i]->gpuDiff,0,sizeof(double)*m_blobs[i]->size());
		CUDA_ERROR(c);
	}
	return 0;
}

int CNET::save(const char* path)
{
	FILE* fp=NULL;
	fp=fopen(path,"wb");
	if(fp==NULL)
		return PK_FAIL;
	for(int i=0;i<m_layers.size();i++)
	{
		if(m_layers[i]->m_weightLen!=0)
			m_layers[i]->save(fp);
	}
	fclose(fp);
	return PK_SUCCESS;
}

int CNET::load(const char* path)
{
	FILE* fp=NULL;
	fp=fopen(path,"rb");
	if(fp==NULL)
		return PK_NOT_FILE;
	for(int i=0;i<m_layers.size();i++)
	{
		if(m_layers[i]->m_weightLen!=0)
			m_layers[i]->load(fp);
	}
	fclose(fp);
	return PK_SUCCESS;
}

int CNET::appendInput(LayerParam& param,int layerID)
{
	int nRet=0;
	for(int i=0;i<param.inputName.size();i++)
	{
		//判断对应的Blob是否存在，不存在就返回错误
		const std::string blobName=param.inputName[i];
		if(m_blobNameToID.find(blobName)==m_blobNameToID.end())
		{
			nRet=-1;
			break;
		}
		int blobID=m_blobNameToID.find(blobName)->second;
		m_forwardInputVec[layerID].push_back(m_blobs[blobID]);
	}
	return nRet;
}

int CNET::appendOutput(LayerParam& param,int layerID)
{
	for(int i=0;i<param.outputName.size();i++)
	{
		//判断对应的Blob是否存在，是就引用原来的Blob
		const std::string blobName=param.outputName[i];
		if(m_blobNameToID.find(blobName)!=m_blobNameToID.end())
		{
			m_forwardOutputVec[layerID].push_back(m_blobs[m_blobNameToID[blobName]]);
			continue;
		}

		//不存在说明是新的输出
		Blob<precision>* tmpBlob=new Blob<precision>();
		m_blobs.push_back(tmpBlob);
		m_blobNameToID[blobName]=m_blobs.size()-1;
		m_forwardOutputVec[layerID].push_back(tmpBlob);
	}
	return 0;
}