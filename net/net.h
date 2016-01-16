#pragma once
#include"..\\net\\layers\\LayerBase.h"

#include<vector>
#include<map>

class CNET
{
public:
	CNET(void);
	virtual ~CNET(void);
	int init(std::vector<LayerParam>& layerParams);
	void setPhase(PHASE phase){m_phase=phase;};

	//分享数据给另一个网络
	int shareDataTo(CNET& dest);
	//一次网络传递，包括正向和反向,返回损失值
	precision step();
	int updateWeights(float momentum,float lr);
	
	precision computeNumericalGradient();

	Blob<precision>* getOutput(std::string blobName)
	{
		std::map<std::string,int>::iterator it=m_blobNameToID.find(blobName);
		if( it == m_blobNameToID.end())
			return NULL;
		return m_blobs[it->second];
	}

	Blob<precision>* getAccuracy()
	{
		return getOutput("ACCURACY");
	}

	CLayer* getLayer(std::string layerName)
	{
		std::map<std::string,int>::iterator it=m_layerNameToID.find(layerName);
		if( it == m_layerNameToID.end())
			return NULL;
		return m_layers[it->second];
	}

	int save(const char* path);
	int load(const char* path);

private:
	PHASE m_phase;

	std::vector<std::vector<Blob<precision>* > > m_forwardInputVec;
	std::vector<std::vector<Blob<precision>* > > m_forwardOutputVec;
	std::vector<std::vector<bool> >              m_layerNeedBack;
	std::vector<Blob<precision>* > m_blobs;
	std::map<std::string,int> m_blobNameToID;
	std::map<std::string,int> m_layerNameToID;
	std::vector<CLayer*> m_layers;

	int feedforward(precision* cost);
	int backpropagation();
	int clearDiff();

	int appendInput(LayerParam& param,int layerID);
	int appendOutput(LayerParam& param,int layerID);
};
