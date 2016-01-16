#include "DataLayerBase.h"
#include "../../Common/PKMatFunctions.h"
#include"../../Common/imgProc/PKImageFunctions.h"
#include<io.h>
#include<fstream>

CDataLayerBase::CDataLayerBase(void)
{
	m_curDataIndex=0;
	m_param.dataParam.batchSize=100;
	m_param.dataParam.dataDim=32;
	m_param.dataParam.datasetType=3;
	m_param.dataParam.geneMeanImgMode=0;
	m_param.dataParam.bGray=false;
}

CDataLayerBase::~CDataLayerBase(void)
{
}

int CDataLayerBase::setup(std::vector<Blob<precision>*>& input,std::vector<Blob<precision>*>& output)
{
	int nRet=readDatas();
	if(nRet!=NET_SUCCESS)
		return nRet;
	if(output.empty()||output.size()>2)
		return NET_OUTPUT_SETTING_ERROR;
	DataParam dataParam=m_param.dataParam;
	int nChannel=dataParam.bGray?1:3;

	if(dataParam.datasetType== 1 )
		nChannel=1;
	if(dataParam.datasetType== 2 )
		nChannel=3;

	if(output.size()<=2)
	{
		output[0]->create(dataParam.batchSize,nChannel,dataParam.dataDim,dataParam.dataDim);
		output[1]->create(dataParam.batchSize,1,1,1);
	}
	else
	{
		output[0]->create(dataParam.batchSize,nChannel,dataParam.dataDim,dataParam.dataDim);
	}

	return NET_SUCCESS;
}

int CDataLayerBase::readFlippedInteger(FILE *fp)
{   
	int ret = NET_SUCCESS;    
	BYTE *temp;   
	temp = (BYTE*)(&ret);  
	fread(&temp[3], sizeof(BYTE), 1, fp);   
	fread(&temp[2], sizeof(BYTE), 1, fp);   
	fread(&temp[1], sizeof(BYTE), 1, fp); 
	fread(&temp[0], sizeof(BYTE), 1, fp); 
	return ret; 
}

int CDataLayerBase::readMNIST(const char* imagesPath,const char* labelsPath)
{
	FILE *fp=fopen(imagesPath,"rb+");
	FILE* fp2=fopen(labelsPath,"rb+");
	if(fp==NULL||fp2==NULL)
		return NET_FILE_NOT_EXIST;

	int m=readFlippedInteger(fp);
	int number=readFlippedInteger(fp);
	int row=readFlippedInteger(fp);
	int col=readFlippedInteger(fp);
	
	m_datas.Resize(number,col*row,1,CpkMat::DATA_BYTE);
	BYTE* buffer=m_datas.GetData<BYTE>();
	fread(buffer,1,row*col*number,fp);
	fclose(fp);

	m=readFlippedInteger(fp2);
	number=readFlippedInteger(fp2);
	
	for(int j=0;j<number;j++)
	{
		char label;
		fread(&label,1,sizeof(char),fp2);
		int dd=label;
		m_labels.push_back(dd);
	}
	fclose(fp2);

	return NET_SUCCESS;
}
int CDataLayerBase::readCIFAR(const char* imagesPath)
{
	char dataPath[250]={0};
	BYTE buffer[32*32]={0};
	int number=0;
	if(m_param.phase == TRAIN)
		number=5;
	else
		number=1;

	m_datas.Resize(number*10000,32*32,3,CpkMat::DATA_BYTE);
	int index=0;
	for(int i=0;i<number;i++)
	{
		if(number==1)
			sprintf(dataPath,"%s\\test_batch.bin",imagesPath);
		else
			sprintf(dataPath,"%s\\data_batch_%d.bin",imagesPath,i+1);
		FILE *fp=fopen(dataPath,"rb+");
		if(fp==NULL)
		{
			printf("%s 文件不存在\n",dataPath);
			return NET_FILE_NOT_EXIST;
		}
		for(int j=0;j<10000;j++,index++)
		{
			char label;
			fread(&label,1,sizeof(char),fp);
			int dd=label;
			m_labels.push_back(dd);

			fread(buffer,1,32*32,fp);
			CpkMat d(32,32,1,CpkMat::DATA_BYTE,buffer);
			BYTE* pSrc=m_datas.GetData<BYTE>(index);
			memcpy(pSrc,d.GetData<BYTE>(),32*32*sizeof(BYTE));

			fread(buffer,1,32*32,fp);
			d.Resize(32,32,1,CpkMat::DATA_BYTE,buffer);
			memcpy(pSrc+sizeof(BYTE)*m_datas.Col,d.GetData<BYTE>(),32*32*sizeof(BYTE));

			fread(buffer,1,32*32,fp);
			d.Resize(32,32,1,CpkMat::DATA_BYTE,buffer);
			memcpy(pSrc+sizeof(BYTE)*m_datas.Col*2,d.GetData<BYTE>(),32*32*sizeof(BYTE));
		}
		fclose(fp);
	}
	return NET_SUCCESS;
}
int CDataLayerBase::readFiles(const char* dataDirect,bool bGray)
{
	std::vector<std::string> directs;
	std::vector<int> directLabels;

	std::ifstream in(dataDirect);
	if(!in.is_open())
	{
		printf("%s 不存在!\n",dataDirect);
		return NET_FILE_NOT_EXIST;
	}
	size_t npos=m_param.dataParam.dataPath.rfind('/');
	if(npos == std::string::npos)
		npos=m_param.dataParam.dataPath.rfind('\\');
	std::string directPath;
	int label;
	while (!in.eof())
	{
		in >> directPath;
		in >> label;
		std::string tmpPath=m_param.dataParam.dataPath.substr(0,npos+1);
		tmpPath+=directPath;
		if(_access(tmpPath.c_str(),0) == -1)
		{
			printf("%s 不存在!",tmpPath.c_str());
			in.close();
			return PK_OPEN_FILE_ERROR;
		}
		directs.push_back(tmpPath);
		directLabels.push_back(label);
	 }      
	in.close();

	int index=0,total=0;
	int nRet=PK_SUCCESS;

	std::vector<std::string> files;
	const char* extensions="bmp;jpg;jpeg";
	for(int i=0;i<directs.size();i++)
	{
		CTools::findDirectsOrFiles(directs[i].c_str(),files,extensions);
		total+=files.size();
		files.clear();
	}

	int dataDim=m_param.dataParam.dataDim;
	for(int i=0;i<directs.size();i++)
	{
		std::vector<std::string> files;
		CTools::findDirectsOrFiles(directs[i].c_str(),files,extensions);
		for(int j=0;j<files.size();j++,index++)
		{
			CpkMat tmp;
			nRet=pk::imread(files[j].c_str(),tmp);
			if(nRet != PK_SUCCESS)
				break;
			nRet=pk::zoomMidImage(tmp,dataDim,pk::BILINEAR);
			if(nRet != PK_SUCCESS)
				break;
			if(tmp.Depth == 1 ||bGray)
			{
				if(m_datas.Empty())
					m_datas.Resize(total,dataDim*dataDim,1,CpkMat::DATA_BYTE);

				nRet=pk::ChangeImageFormat(tmp,tmp,pk::BGR2GRAY);
				if(nRet != PK_SUCCESS)
					break;
				nRet=m_datas.setData(tmp.RowVector(),index,index+1,0,tmp.lineSize);
				if(nRet != PK_SUCCESS)
					break;
			}
			else
			{
				if(m_datas.Empty())
					m_datas.Resize(total,dataDim*dataDim,3,CpkMat::DATA_BYTE);
				std::vector<CpkMat> channels;
				pk::Split(tmp,channels);
				BYTE* pSrc=m_datas.GetData<BYTE>(index);
				memcpy(pSrc,channels[2].GetData<BYTE>(),dataDim*dataDim*sizeof(BYTE));
				memcpy(pSrc+sizeof(BYTE)*m_datas.Col,channels[1].GetData<BYTE>(),dataDim*dataDim*sizeof(BYTE));
				memcpy(pSrc+sizeof(BYTE)*m_datas.Col*2,channels[0].GetData<BYTE>(),dataDim*dataDim*sizeof(BYTE));
			}
			m_labels.push_back(directLabels[i]);
		}
		if(nRet != NET_SUCCESS)
			break;
	}

	return nRet;
}

int CDataLayerBase::geneMeanImg(CpkMat& meanImg)
{
	int channel=m_datas.Depth;
	if(meanImg.Empty())
		meanImg.Resize(1,m_datas.Col,channel,CpkMat::DATA_DOUBLE);
	BYTE* pSrcData=m_datas.GetData<BYTE>();
	for(int i=0;i<m_datas.Row;i++)
	{
		CpkMat tmp(1,m_datas.Col,channel,CpkMat::DATA_BYTE,pSrcData+i*m_datas.lineSize);
		tmp.Resize(CpkMat::DATA_DOUBLE);
		tmp/=255;
		meanImg+=tmp;
	}
	meanImg/=m_datas.Row;
	return NET_SUCCESS;
}

int CDataLayerBase::readDatas()
{
	int nRet=NET_SUCCESS;
	switch(m_param.dataParam.datasetType)
	{
	case 1:
		{
			char imgPath[256]={0};
			char labelPath[256]={0};
			if(m_param.phase == TRAIN)
			{
				sprintf(imgPath,"%s/train-images-idx3-ubyte",m_param.dataParam.dataPath.c_str());
				sprintf(labelPath,"%s/train-labels-idx1-ubyte",m_param.dataParam.dataPath.c_str());
			}
			else
			{
				sprintf(imgPath,"%s/t10k-images-idx3-ubyte",m_param.dataParam.dataPath.c_str());
				sprintf(labelPath,"%s/t10k-labels-idx1-ubyte",m_param.dataParam.dataPath.c_str());
			}
			printf("图片集所在路径:%s\n",imgPath);
			printf("标签集所在路径:%s\n",labelPath);
			nRet=readMNIST(imgPath,labelPath);
		}
		break;
	case 2:
		nRet=readCIFAR(m_param.dataParam.dataPath.c_str());
		break;
	case 3:
			nRet=readFiles(m_param.dataParam.dataPath.c_str(),m_param.dataParam.bGray);
		break;
	default:
		nRet=NET_NOT_ALLOW_OPERATOR;
	}

	if(nRet==NET_SUCCESS&&m_param.dataParam.geneMeanImgMode>0)
	{
		if(m_param.phase!=0&&pk::load(m_param.dataParam.meanDataPath.c_str(),m_meanImg)==PK_SUCCESS)
			return NET_SUCCESS;
		else
			return NET_FILE_NOT_EXIST;

		if(m_param.dataParam.geneMeanImgMode==1
			&&_access(m_param.dataParam.meanDataPath.c_str(),0) == 0
			&&pk::load(m_param.dataParam.meanDataPath.c_str(),m_meanImg)==NET_SUCCESS)
			return NET_SUCCESS;

		nRet=geneMeanImg(m_meanImg);
		if(nRet!=NET_SUCCESS)
			return nRet;
		nRet=pk::save(m_param.dataParam.meanDataPath.c_str(),m_meanImg);
	}

	return nRet;
}

precision CDataLayerBase::feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	if(m_param.phase == TRAIN&&m_shuffleIndex.Empty())
		shuffle();

	if(m_datas.Empty())
	{
		int nRet=readDatas();
		if(nRet!=NET_SUCCESS)
			return nRet;
	}

	if(m_curDataIndex>=m_datas.Row)
	{
		if(m_param.phase == TRAIN)
			shuffle();
		m_curDataIndex=0;
	}

	if(m_batchDatas.Empty()||m_batchDatas.Row!=m_param.dataParam.batchSize)
	{
		m_batchDatas.Resize(m_param.dataParam.batchSize,m_datas.lineSize,1,CpkMat::DATA_DOUBLE);
		m_batchLabels=new precision[m_param.dataParam.batchSize];
	}
	for(int i=0;i<m_param.dataParam.batchSize&&i+m_curDataIndex<m_datas.Row;i++)
	{
		CpkMat tmp;
		int index=m_curDataIndex+i;
		if(!m_shuffleIndex.Empty())
			index=m_shuffleIndex.at<int>(0,m_curDataIndex+i);
		m_datas.GetData(tmp,index,index+1,0,m_datas.Col);

		tmp.Resize(CpkMat::DATA_DOUBLE);
		tmp=tmp/255;
		if(!m_meanImg.Empty())
			tmp-=m_meanImg;

		m_batchDatas.setData(tmp.RowVector(),i,i+1,0,m_datas.lineSize);
		m_batchLabels[i]=m_labels[index];
	}

	tops[0]->copyFromCpuData(m_batchDatas.GetData<double>(),m_batchDatas.Row*m_batchDatas.Col);

	tops[1]->copyFromCpuData(m_batchLabels,m_batchDatas.Row);

	if(m_param.dataParam.bMove)
		m_curDataIndex+=m_param.dataParam.batchSize;

	return 0.0;
}
int CDataLayerBase::shuffle()
{
	m_shuffleIndex=pk::randperm(m_datas.Row);
	if(m_shuffleIndex.Empty())
		return PK_FAIL;
	return PK_SUCCESS;
}
int CDataLayerBase::setImgProcType(int type,float param1,float param2)
{
	_ImgProcMess imgType={0};
	imgType.type=type;
	imgType.param1=param1;
	imgType.param2=param2;
	m_imgProc.push_back(imgType);
	return PK_SUCCESS;
}