#ifndef LAYER_PARAM_H
#define LAYER_PARAM_H
#include<string>
#include<vector>

#define MAX_POOL 0
#define AVG_POOL 1
#define RAND_POOL 2

#define ACROSS_CHANNELS 0
#define WITHIN_CHANNEL  1

enum PHASE{TRAIN=0,TEST,EMPTY};

struct DataParam
{
	DataParam():batchSize(200),dataDim(28),datasetType(1),
		geneMeanImgMode(0),bGray(false),bMove(true){}
	int    batchSize;
	bool   bMove;    //是否按batchSize遍历数据集，0为固定最先batchSize个，用于梯度校验
	int    dataDim;
	int    datasetType;// dataType:1 is read MNIST,2 is read CIFAR,3 is image files
	int    geneMeanImgMode;// geneMeanImgMode: 0 is no create meanImg.1 is if meanImg no existence that create. 2 is create and cover old data
	bool   bGray;// bGary: if bGray is true that Original(RGB) change to gray
    // path:Original data Direct
	// meanImgPath:save or load meanImg data path
	std::string  dataPath;
	std::string  meanDataPath;
};

typedef struct _convParam
{
	int kernelDim;
}ConvParam;

typedef struct _poolParam
{
	int kernelDim;
	int poolType;
}PoolParam;

typedef struct _dropoutParam
{
	float dropoutRate;
}DropoutParam;

typedef struct _LRNParam
{
	int    normRegionType;
	int    localSize;
	float  alpha;
	float  beat;
}LRNParam;

struct LayerParam
{
	LayerParam():curNumFeature(0),biasLearnRatio(1.0),weightLearnRatio(1.0)
	{
		propagateDown=true;
		phase=EMPTY;
	}
		
	DataParam dataParam;
	ConvParam convParam;
	PoolParam poolParam;
	DropoutParam dropoutParam;
	LRNParam  lrnParam;

	int     curNumFeature;
	float   lambda;
	float   biasLearnRatio;
	float   weightLearnRatio;
	bool    propagateDown;
	PHASE  phase;      //0 is train and 1 is test
	std::string name;
	std::string typeName;
	std::vector<std::string> inputName;
	std::vector<std::string> outputName;
};

#endif