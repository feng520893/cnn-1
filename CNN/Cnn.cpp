#include "Cnn.h"
#include <assert.h>
#include<Windows.h>

#include"../Common/DLerror.h"
#include"..\\common\\pkmatFunctions.h"

#pragma comment( lib,"winmm.lib")

CCNN::CCNN(void)
{
	m_sgd.alpha=0.05;
	m_sgd.epoches=30;
	m_sgd.minibatch=200;
	m_sgd.momentum=0.9;
	m_activeType=NL_NONE;
}

CCNN::~CCNN(void)
{
	for(int i=0;i<m_cnn.size();i++)
		delete m_cnn[i];
}

void CCNN::setParam(MinSGD& sgd,int activeType,int runMode)
{
	m_sgd=sgd;
	m_activeType=activeType;
	m_runMode=runMode;
}

const MinSGD& CCNN::getMinSGDInfo()
{
	return m_sgd;
}

int CCNN::init(std::vector<CLayer*>& cnnLayers)
{
	m_cnn=cnnLayers;
	DL_ASSER(!m_cnn.empty());

	return PK_SUCCESS;
}

int CCNN::cnnTrain(Data&datas,std::vector<int>& labels,const char* savePath,int maxSaveCount)
{
	if(datas.Row==0)
	{
		printf("训练集为空");
		return PK_FAIL;
	}

	if(datas.Row<m_sgd.minibatch)
	{
		printf("小样本的数量(%d)大于总数(%d)",m_sgd.minibatch,datas.Row);
		return PK_FAIL;
	}

	int it = 0;
	int momIncrease=20;
	double mom=0.5;
	double alpha = m_sgd.alpha;
	int dataSize=labels.size();

	std::vector<int> nLocalLabel;
	int nChannel=m_cnn[0]->m_inputNumFeature;
	int imageSize=datas.Col;
	double* nLocalData=(double*)malloc(sizeof(double)*imageSize*nChannel*m_sgd.minibatch);

	double* gpuData=NULL;

	if(m_runMode==PK_CNN_GPU_CUDA_RUN)
	{
		cudaError_t cudaStat;
		cudaStat=cudaMalloc((void**)&gpuData,sizeof(double)*imageSize*nChannel*m_sgd.minibatch);
		CUDA_ERROR(cudaStat);
	}
	
	nLocalLabel.resize(m_sgd.minibatch);
	std::vector<double> p;
	double cost=0.0;
	double allCost=0.0;
	int count=0;
	int saveCount=0;
	for(int e = 0;e<m_sgd.epoches;e++)
	{
		DWORD time=timeGetTime();
		CpkMat rp = pk::randperm(dataSize);
		for(int s=0;s<dataSize-m_sgd.minibatch;s+=m_sgd.minibatch)
		{
			if(++it == momIncrease)
				mom = m_sgd.momentum;
			CpkMat tmp;
			rp.GetData(tmp,0,1,s,s+m_sgd.minibatch);
			int* pdata=tmp.GetData<int>();
			double* pSrcData=datas.GetData<double>();
			for(int i=0;i<m_sgd.minibatch;i++,pdata++)
			{
				memcpy(nLocalData+i*imageSize*nChannel,pSrcData+(*pdata)*imageSize*nChannel,sizeof(double)*imageSize*nChannel);
				nLocalLabel[i]=labels[*pdata];
			}

			if(m_runMode==PK_CNN_GPU_CUDA_RUN)
			{
				cudaError_t cudaStat;
				cudaStat=cudaMemcpy(gpuData,nLocalData,sizeof(double)*imageSize*nChannel*m_sgd.minibatch,cudaMemcpyHostToDevice);
				CUDA_ERROR(cudaStat);
			}
			else
				gpuData=nLocalData;

			cost=cnnCost(p,gpuData,nLocalLabel);
			updateWeights();
			allCost+=cost;
			++count;
			printf("单次训练进度:%.2lf%%",100.0*s/dataSize);
			for(int g=0;g<20;g++)
				printf("\b");
		}
		printf("第%d次迭代误差:%lf 用时%dS\n",e+1,allCost/count,(timeGetTime()-time)/1000);
		allCost=0;
		count=0;

		++saveCount;
		if(maxSaveCount!=0&&saveCount>=maxSaveCount)
		{
			save(savePath);
			saveCount=0;
		}
	}
	free(nLocalData);
	if(m_runMode==PK_CNN_GPU_CUDA_RUN)
		cudaFree(gpuData);
	return PK_SUCCESS;
}

int CCNN::cnnRun(std::vector<double>&pred,Data&datas)
{
	if(datas.Row==0||datas.Col==0)
	{	
		printf("测试集为空");
		return PK_FAIL;
	}

	int dataSize=datas.Row;
	int imageSize=datas.Col;
	int nChannel=m_cnn[0]->m_inputNumFeature;

	double* gpuData=NULL;

	if(m_runMode==PK_CNN_GPU_CUDA_RUN)
	{
		cudaError_t cudaStat;
		cudaStat=cudaMalloc((void**)&gpuData,sizeof(double)*imageSize*nChannel*dataSize);
		CUDA_ERROR(cudaStat);
	}

	double* nLocalData=(double*)malloc(sizeof(double)*imageSize*nChannel*dataSize);
	for(int i=0;i<dataSize;i++)
		memcpy(nLocalData+i*imageSize*nChannel,datas.GetData<double>()+i*imageSize*nChannel,sizeof(double)*imageSize*nChannel);

	if(m_runMode==PK_CNN_GPU_CUDA_RUN)
	{
		cudaError_t cudaStat;
		cudaStat=cudaMemcpy(gpuData,nLocalData,sizeof(double)*imageSize*nChannel*dataSize, cudaMemcpyHostToDevice);
		CUDA_ERROR(cudaStat);
		free(nLocalData);
	}
	else
		gpuData=nLocalData;

	std::vector<int> empLaber;
	cnnCost(pred,gpuData,empLaber,true);

	if(m_runMode==PK_CNN_CPU_RUN)
		free(nLocalData);
	return PK_SUCCESS;
}


double CCNN::computeNumericalGradient(Data&srcDatas,std::vector<int>& srcLabels)
{
	Data datas;
	std::vector<int> labels;
	if(srcLabels.size()>2)
	{
		for(int i=0;i<2;i++)
			labels.push_back(srcLabels[i]);
		srcDatas.copyTo(datas,2,srcDatas.Col,srcDatas.GetType());
	}
	else
	{
		datas=srcDatas;
		labels=srcLabels;
	}

	int dataSize=datas.Row;
	int imageSize=datas.Col;
	int nChannel=m_cnn[0]->m_inputNumFeature;

	double* gpuData=NULL;
	if(m_runMode==PK_CNN_GPU_CUDA_RUN)
	{
		cudaError_t cudaStat=cudaMalloc((void**)&gpuData,sizeof(double)*imageSize*nChannel*dataSize);
		CUDA_ERROR(cudaStat);
	}

	double* nLocalData=(double*)malloc(sizeof(double)*imageSize*nChannel*dataSize);
	for(int i=0;i<dataSize;i++)
		memcpy(nLocalData+i*imageSize*nChannel,datas.GetData<double>()+i*imageSize*nChannel,sizeof(double)*imageSize*nChannel);

	if(m_runMode==PK_CNN_GPU_CUDA_RUN)
	{
		cudaError_t cudaStat=cudaMemcpy(gpuData,nLocalData,sizeof(double)*imageSize*nChannel*dataSize, cudaMemcpyHostToDevice);
		free(nLocalData);
		CUDA_ERROR(cudaStat);
	}
	else
		gpuData=nLocalData;

	std::vector<double> p;
	cnnCost(p,gpuData,labels);

	unsigned int size=0;
	for(int i=0;i<m_cnn.size();i++)
		size+=m_cnn[i]->m_weightLen+m_cnn[i]->m_curNumFeature;

	double* numgrad=new double[size]; 
	//拷贝梯度差
	double* grad=new double[size];
	unsigned int index=0;
	for(int i=0;i<m_cnn.size();i++)
	{
		if(m_cnn[i]->m_weightLen==0)
			continue;
		m_cnn[i]->getWeightsGrad(grad+index);
		index+=m_cnn[i]->m_weightLen;
		m_cnn[i]->getBiasGrad(grad+index);
		index+=m_cnn[i]->m_curNumFeature;
	}

	double epsilon = 1e-4;

	int index2=0;
	for(int i=0;i<m_cnn.size();i++)
	{
		int weightSize=m_cnn[i]->m_weightLen;

		if(weightSize==0)
			continue;

		int biasSize=m_cnn[i]->m_curNumFeature;
		for(int j=0;j<weightSize;j++)
		{
			double oldT = m_cnn[i]->getWeightValue(j);
			m_cnn[i]->setWeightValue(j,oldT+epsilon);
			double pos = cnnCost(p,gpuData,labels);
			m_cnn[i]->setWeightValue(j,oldT-epsilon);
			double neg = cnnCost(p,gpuData,labels);
			numgrad[index2++]= (pos-neg)/(2*epsilon);
			m_cnn[i]->setWeightValue(j,oldT);
		}
		for(int j=0;j<biasSize;j++)
		{
			double oldT = m_cnn[i]->getBiasValue(j);
			m_cnn[i]->setBiasValue(j,oldT+epsilon);
			double pos = cnnCost(p,gpuData,labels);
			m_cnn[i]->setBiasValue(j,oldT-epsilon);
			double neg = cnnCost(p,gpuData,labels);
			numgrad[index2++]= (pos-neg)/(2*epsilon);
			m_cnn[i]->setBiasValue(j,oldT);
		}
	}

	for(int i=0;i<index;i++)
	{
		if(i<52)
			printf("%.6lf   %.6lf\n",numgrad[i],grad[i]);
		else if(i<256)
			printf("%.6lf   %.6lf\n",numgrad[i],grad[i]);
		else if(i<856)
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

	if(m_runMode==PK_CNN_CPU_RUN)
		free(nLocalData);
    else if(m_runMode==PK_CNN_GPU_CUDA_RUN)
		cudaFree(gpuData);

	delete [] numgrad;
	delete [] grad;

	return sum1/sum2;
}

int CCNN::feedforward(std::vector<double>&predMat,double*data,bool bPred)
{
	static DLparam params;
	params.activeType=m_activeType;
	params.pred=bPred;

	if(bPred&&params.predData==NULL)
	{
		params.predData=new double[m_sgd.minibatch];
		DL_ASSER(params.predData!=NULL);
	}

	for(int i=0;i<m_cnn.size();i++)
	{
		if(i==0)
			m_cnn[i]->feedforward(data,params);
		else
			m_cnn[i]->feedforward(m_cnn[i-1]->getOutPut(),params);
	}
	if(bPred)
	{
		for(int i=0;i<m_sgd.minibatch;i++)
			predMat.push_back(params.predData[i]);
	}
	return 0;
}

double CCNN::getCost(std::vector<int>& labels)
{
	DLparam params(labels);
	double res=0.0;
	for(int i=0;i<m_cnn.size();i++)
		res+=m_cnn[i]->getCost(params);
	return res;
}

int CCNN::backpropagation(std::vector<int>& labels)
{
	DLparam params;
	params.labels=labels;
	params.activeType=m_activeType;

	for(int i=m_cnn.size()-1;i>0;i--)
		m_cnn[i]->backpropagation(m_cnn[i-1]->getDelta(),params);
	m_cnn[0]->backpropagation(NULL,params);

	return 0;
}

double CCNN::cnnCost(std::vector<double>&predMat,double*data,std::vector<int>& labels,bool bPred)
{
	feedforward(predMat,data,bPred);
	if(bPred)
		return 0;

	double result=getCost(labels);

	backpropagation(labels);

	grads(data);

	return result;
}

int CCNN::grads(double* srcData)
{
	for(int i=1;i<m_cnn.size();i++)
		m_cnn[i]->getGrad(m_cnn[i-1]->getOutPut());
	m_cnn[0]->getGrad(srcData);

	return PK_SUCCESS;
}

int CCNN::updateWeights()
{
	for(int i=0;i<m_cnn.size();i++)
		m_cnn[i]->updateWeight(m_sgd.momentum,m_sgd.alpha);
	return PK_SUCCESS;
}

int CCNN::save(const char* path)
{
	FILE* fp=NULL;
	fp=fopen(path,"wb");
	if(fp==NULL)
		return PK_FAIL;
	for(int i=0;i<m_cnn.size();i++)
	{
		if(m_cnn[i]->m_weightLen!=0)
			m_cnn[i]->save(fp);
	}
	fclose(fp);
	return PK_SUCCESS;
}

int CCNN::load(const char* path)
{
	FILE* fp=NULL;
	fp=fopen(path,"rb");
	if(fp==NULL)
		return PK_NOT_FILE;
	for(int i=0;i<m_cnn.size();i++)
	{
		if(m_cnn[i]->m_weightLen!=0)
			m_cnn[i]->load(fp);
	}
	fclose(fp);
	return PK_SUCCESS;
}