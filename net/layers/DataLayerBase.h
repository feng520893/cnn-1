#pragma once
#include <vector>
#include "../../Common/CpkMat.h"
#include"LayerBase.h"
#include<string>
class CDataLayerBase : public CLayer
{
public:
	CDataLayerBase(void);
	~CDataLayerBase(void);

	int readDatas();

	int setImgProcType(int type,float param1=0.0,float param2=0.0);

	precision feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops);

	int setup(std::vector<Blob<precision>*>& input,std::vector<Blob<precision>*>& output);
private:
	int readFlippedInteger(FILE *fp); //read mnist help function
	int readMNIST(const char* imagesPath,const char* labelsPath);
	int readCIFAR(const char* path);
	int readFiles(const char* path,bool bGray);

	int shuffle();  //¥Ú¬“À≥–Ú
	int geneMeanImg(CpkMat& meanImg);

	struct _ImgProcMess
	{
		int type;
		float param1,param2;
	};
	CpkMat m_datas,m_batchDatas,m_shuffleIndex,m_meanImg;
	std::vector<_ImgProcMess> m_imgProc;
	std::vector<int>  m_labels;
	precision*  m_batchLabels;
	unsigned int      m_curDataIndex;
};
