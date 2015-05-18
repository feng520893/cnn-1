#include "Cnn.h"
#include <assert.h>

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
}

void CCNN::setParam(MinSGD& sgd,float dropRate,int activeType)
{
	m_sgd=sgd;
	m_dropRate=dropRate;
	m_activeType=activeType;
}

const MinSGD& CCNN::getMinSGDInfo()
{
	return m_sgd;
}

int CCNN::init(std::vector<CConvLayer>& convLayers,std::vector<CFullLayer>&fullLayers,CSoftMaxLayer& sfm,int dataDim,short dataChannel,int batch)
{
	if(convLayers.empty()||NL_NONE==m_activeType)
		return PK_CNN_NOT_INIT;
	
	for(int i=0;i<convLayers.size();i++)
	{
		if(i==0)
			convLayers[i].m_inputNumFeature=dataChannel;
		else
			convLayers[i].m_inputNumFeature=convLayers[i-1].m_curNumFeature;
		dataDim=dataDim-convLayers[i].m_maskDim+1;
		assert(dataDim%convLayers[i].m_poolArea==0);
		
		convLayers[i].m_convDim=dataDim;
		dataDim/=convLayers[i].m_poolArea;
		
		convLayers[i].batch=batch;
		
	}
	if(!fullLayers.empty())
	{
		
		for(int i=0;i<fullLayers.size();i++)
		{
			if(i==0)
				fullLayers[0].m_inputNumFeature=dataDim*dataDim*convLayers[convLayers.size()-1].m_curNumFeature;
			else
				fullLayers[i].m_inputNumFeature=fullLayers[i-1].m_curNumFeature;
			fullLayers[i].batch=batch;
			fullLayers[i].m_rate=m_dropRate;
		}
		sfm.m_inputNumFeature=fullLayers[fullLayers.size()-1].m_curNumFeature;
	}
	else
		sfm.m_inputNumFeature=dataDim*dataDim*convLayers[convLayers.size()-1].m_curNumFeature;
	sfm.batch=batch;


	m_convLayers=convLayers;
	m_fullLayers=fullLayers;
	m_sfm=sfm;
	for(int i=0;i<m_convLayers.size();i++)
		m_convLayers[i].initMem();
	for(int i=0;i<fullLayers.size();i++)
		m_fullLayers[i].initMem();
	m_sfm.initMem();

	return PK_SUCCESS;
}

int CCNN::cnnTrain(Data&datas,std::vector<int>& labels)
{
	if(datas.Row==0)
	{
		printf("训练集为空");
		return PK_FAIL;
	}

	if(datas.Row%m_sgd.minibatch!=0)
	{
		printf("小样本的数量(%d)不能被总数(%d)的倍数",m_sgd.minibatch,datas.Row);
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
	int nChannel=m_convLayers[0].m_inputNumFeature;
	int imageSize=datas.Col;
	double* nLocalData=(double*)malloc(sizeof(double)*imageSize*nChannel*m_sgd.minibatch);

	double* gpuData=NULL;
	cudaError_t cudaStat;
	cudaStat=cudaMalloc((void**)&gpuData,sizeof(double)*imageSize*nChannel*m_sgd.minibatch);
	if(cudaStat!=cudaSuccess)
	{
		printf("device memory cudaMalloc failed\n");
		return -1;
	}
	
	nLocalLabel.resize(m_sgd.minibatch);
	std::vector<double> p;
	double cost=0.0;
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

			cudaStat=cudaMemcpy(gpuData,nLocalData,sizeof(double)*imageSize*nChannel*m_sgd.minibatch,cudaMemcpyHostToDevice);
			if(cudaStat!=cudaSuccess)
			{
				printf("device memory cudaMemcpy failed\n");
				return -2;
			}

			cost=cnnCost(p,gpuData,nLocalLabel);
			updateWeights();
			printf("单次训练进度:%.2lf%%",100.0*s/dataSize);
			for(int g=0;g<20;g++)
				printf("\b");
		}
		printf("第%d次迭代误差:%lf 用时%dS\n",e+1,cost,(timeGetTime()-time)/1000);
	}
	free(nLocalData);
	GPU_FREE(gpuData);
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
	int nChannel=m_convLayers[0].m_inputNumFeature;

	double* gpuData=NULL;
	cudaError_t cudaStat;
	cudaStat=cudaMalloc((void**)&gpuData,sizeof(double)*imageSize*nChannel*dataSize);
	if(cudaStat!=cudaSuccess)
	{
		printf("device memory cudaMalloc failed\n");
		return -1;
	}

	double* nLocalData=(double*)malloc(sizeof(double)*imageSize*nChannel*dataSize);
	for(int i=0;i<dataSize;i++)
		memcpy(nLocalData+i*imageSize*nChannel,datas.GetData<double>()+i*imageSize*nChannel,sizeof(double)*imageSize*nChannel);

	cudaStat=cudaMemcpy(gpuData,nLocalData,sizeof(double)*imageSize*nChannel*dataSize, cudaMemcpyHostToDevice);
	free(nLocalData);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMemcpy failed\n"); 
		return -3;
	}

	std::vector<int> empLaber;
	cnnCost(pred,gpuData,empLaber,true);
	GPU_FREE(nLocalData);
	return PK_SUCCESS;
}


double CCNN::computeNumericalGradient(Data&srcDatas,std::vector<int>& srcLabels)
{
	Data datas;
	std::vector<int> labels;
	if(labels.size()>2)
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
	int nChannel=m_convLayers[0].m_inputNumFeature;

	double* gpuData=NULL;
	cudaError_t cudaStat;
	cudaStat=cudaMalloc((void**)&gpuData,sizeof(double)*imageSize*m_convLayers[0].m_inputNumFeature*dataSize);
	if(cudaStat!=cudaSuccess)
	{
		printf("device memory cudaMalloc failed\n");
		return -1;
	}

	double* nLocalData=(double*)malloc(sizeof(double)*imageSize*m_convLayers[0].m_inputNumFeature*dataSize);
	for(int i=0;i<dataSize;i++)
		memcpy(nLocalData+i*imageSize*nChannel,datas.GetData<double>()+i*imageSize*nChannel,sizeof(double)*imageSize*nChannel);

	cudaStat=cudaMemcpy(gpuData,nLocalData,sizeof(double)*imageSize*m_convLayers[0].m_inputNumFeature*dataSize, cudaMemcpyHostToDevice);
	free(nLocalData);
	if(cudaStat!=cudaSuccess)
	{
		printf ("device memory cudaMemcpy failed\n"); 
		return -3;
	}

	std::vector<double> p;
	cnnCost(p,gpuData,labels);

	unsigned int size=0;
	for(int i=0;i<m_convLayers.size();i++)
		size+=m_convLayers[i].m_curNumFeature*m_convLayers[i].m_inputNumFeature*m_convLayers[i].m_maskDim*m_convLayers[i].m_maskDim+m_convLayers[i].m_curNumFeature;
	for(int i=0;i<m_fullLayers.size();i++)
		size+=m_fullLayers[i].m_curNumFeature*m_fullLayers[i].m_inputNumFeature+m_fullLayers[i].m_curNumFeature;
	size+=m_sfm.m_curNumFeature*m_sfm.m_inputNumFeature+m_sfm.m_curNumFeature;

	double* numgrad=new double[size]; 
	//拷贝梯度差
	double* grad=new double[size];
	unsigned int index=0;
	for(int i=0;i<m_convLayers.size();i++)
	{
		m_convLayers[i].getWeightsGrad(grad+index);
		index+=m_convLayers[i].m_curNumFeature*m_convLayers[i].m_inputNumFeature*m_convLayers[i].m_maskDim*m_convLayers[i].m_maskDim;
		m_convLayers[i].getBiasGrad(grad+index);
		index+=m_convLayers[i].m_curNumFeature;
	}
	for(int i=0;i<m_fullLayers.size();i++)
	{
		m_fullLayers[i].getWeightsGrad(grad+index);
		index+=m_fullLayers[i].m_curNumFeature*m_fullLayers[i].m_inputNumFeature;
		m_fullLayers[i].getBiasGrad(grad+index);
		index+=m_fullLayers[i].m_curNumFeature;
	}
	m_sfm.getWeightsGrad(grad+index);
	index+=m_sfm.m_curNumFeature*m_sfm.m_inputNumFeature;
	m_sfm.getBiasGrad(grad+index);
	index+=m_sfm.m_curNumFeature;

	double epsilon = 1e-4;

	int index2=0;
	for(int i=0;i<m_convLayers.size();i++)
	{
		int weightSize=m_convLayers[i].m_curNumFeature*m_convLayers[i].m_inputNumFeature*m_convLayers[i].m_maskDim*m_convLayers[i].m_maskDim;
		int biasSize=m_convLayers[i].m_curNumFeature;
		for(int j=0;j<weightSize;j++)
		{
			double oldT = m_convLayers[i].getWeightValue(j);
			m_convLayers[i].setWeightValue(j,oldT+epsilon);
			double pos = cnnCost(p,gpuData,labels);
			m_convLayers[i].setWeightValue(j,oldT-epsilon);
			double neg = cnnCost(p,gpuData,labels);
			numgrad[index2++]= (pos-neg)/(2*epsilon);
			m_convLayers[i].setWeightValue(j,oldT);
		}
		for(int j=0;j<biasSize;j++)
		{
			double oldT = m_convLayers[i].getBiasValue(j);
			m_convLayers[i].setBiasValue(j,oldT+epsilon);
			double pos = cnnCost(p,gpuData,labels);
			m_convLayers[i].setBiasValue(j,oldT-epsilon);
			double neg = cnnCost(p,gpuData,labels);
			numgrad[index2++]= (pos-neg)/(2*epsilon);
			m_convLayers[i].setBiasValue(j,oldT);
		}
	}

	for(int i=0;i<m_fullLayers.size();i++)
	{
		int weightSize=m_fullLayers[i].m_curNumFeature*m_fullLayers[i].m_inputNumFeature;
		int biasSize=m_fullLayers[i].m_curNumFeature;
		for(int j=0;j<weightSize;j++)
		{
			double oldT = m_fullLayers[i].getWeightValue(j);
			m_fullLayers[i].setWeightValue(j,oldT+epsilon);
			double pos = cnnCost(p,gpuData,labels);
			m_fullLayers[i].setWeightValue(j,oldT-epsilon);
			double neg = cnnCost(p,gpuData,labels);
			numgrad[index2++]= (pos-neg)/(2*epsilon);
			m_fullLayers[i].setWeightValue(j,oldT);

			if(numgrad[index2-1]>1||numgrad[index2-1]<-1)
			{
				numgrad[index2++]=0;
			}
		}
		for(int j=0;j<biasSize;j++)
		{
			double oldT = m_fullLayers[i].getBiasValue(j);
			m_fullLayers[i].setBiasValue(j,oldT+epsilon);
			double pos = cnnCost(p,gpuData,labels);
			m_fullLayers[i].setBiasValue(j,oldT-epsilon);
			double neg = cnnCost(p,gpuData,labels);
			numgrad[index2++]= (pos-neg)/(2*epsilon);
			m_fullLayers[i].setBiasValue(j,oldT);
		}
	}

		int weightSize=m_sfm.m_curNumFeature*m_sfm.m_inputNumFeature;
		int biasSize=m_sfm.m_curNumFeature;
		for(int j=0;j<weightSize;j++)
		{
			double oldT = m_sfm.getWeightValue(j);
			m_sfm.setWeightValue(j,oldT+epsilon);
			double pos = cnnCost(p,gpuData,labels);
			m_sfm.setWeightValue(j,oldT-epsilon);
			double neg = cnnCost(p,gpuData,labels);
			numgrad[index2++]= (pos-neg)/(2*epsilon);
			m_sfm.setWeightValue(j,oldT);
		}
		for(int j=0;j<biasSize;j++)
		{
			double oldT = m_sfm.getBiasValue(j);
			m_sfm.setBiasValue(j,oldT+epsilon);
			double pos = cnnCost(p,gpuData,labels);
			m_sfm.setBiasValue(j,oldT-epsilon);
			double neg = cnnCost(p,gpuData,labels);
			numgrad[index2++]= (pos-neg)/(2*epsilon);
			m_sfm.setBiasValue(j,oldT);
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

	GPU_FREE(gpuData);

	delete [] numgrad;
	delete [] grad;

	return sum1/sum2;
}

int CCNN::feedforward(std::vector<double>&predMat,double*data,bool bPred)
{

	for(int i=0;i<m_convLayers.size();i++)
	{
		if(i==0)
			m_convLayers[i].feedforward(data,m_activeType);
		else
			m_convLayers[i].feedforward(m_convLayers[i-1].m_poolData,m_activeType);
	}

	for(int i=0;i<m_fullLayers.size();i++)
	{
		if(i==0)
			m_fullLayers[i].feedforward(m_convLayers[m_convLayers.size()-1].m_poolData,m_activeType,bPred);
		else
			m_fullLayers[i].feedforward(m_fullLayers[i-1].m_fullData,m_activeType,bPred);
	}

	double* pred=NULL;
	if(!m_fullLayers.empty())
		m_sfm.feedforward(&pred,m_fullLayers[m_fullLayers.size()-1].m_fullData,bPred);
	else
		m_sfm.feedforward(&pred,m_convLayers[m_convLayers.size()-1].m_poolData,bPred);
	if(bPred)
	{
		for(int i=0;i<m_sfm.batch;i++)
			predMat.push_back(pred[i]);
	}
	if(pred!=NULL)
		delete [] pred;
	return 0;
}

double CCNN::getCost(std::vector<int>& labels)
{
	double res=0.0;
	for(int i=0;i<m_convLayers.size();i++)
		res+=m_convLayers[i].getCost();
	for(int i=0;i<m_fullLayers.size();i++)
		res+=m_fullLayers[i].getCost();
	return res+m_sfm.getCost(labels);
}

int CCNN::backpropagation(std::vector<int>& labels)
{
	DWORD tt=timeGetTime();
	int topIndex=m_convLayers.size()-1;
	m_sfm.backpropagation(labels);

//	printf("2.1:%d\n",timeGetTime()-tt);

	if(!m_fullLayers.empty())
	{
		m_fullLayers[m_fullLayers.size()-1].backpropagation(m_sfm.m_delta,m_activeType);

//		printf("2.1.1:%d\n",timeGetTime()-tt);

		for(int i=m_fullLayers.size()-2;i>=0;i--)
		{
			m_fullLayers[i].backpropagation(m_fullLayers[i+1].m_delta,m_activeType);
		}
//		printf("2.1.2:%d\n",timeGetTime()-tt);
		cudaMemcpy(m_convLayers[topIndex].m_poolDelta,m_fullLayers[0].m_delta,sizeof(double)*m_fullLayers[0].m_inputNumFeature*m_fullLayers[0].batch,cudaMemcpyDeviceToDevice);
	}
	else
	{
		cudaMemcpy(m_convLayers[topIndex].m_poolDelta,m_sfm.m_delta,sizeof(double)*m_sfm.m_inputNumFeature*m_sfm.batch,cudaMemcpyDeviceToDevice);
	}

//	printf("2.2:%d\n",timeGetTime()-tt);

	for(int i=m_convLayers.size()-1;i>0;i--)
		m_convLayers[i].backpropagation(m_convLayers[i-1].m_poolDelta,m_activeType);
	m_convLayers[0].backpropagation(NULL,m_activeType);

//	printf("2.3:%d\n",timeGetTime()-tt);

	return 0;
}

#include<Windows.h>

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
	DWORD tt=timeGetTime();
	if(!m_fullLayers.empty())
	{
		m_sfm.getGrad(m_fullLayers[m_fullLayers.size()-1].m_fullData);

		for(int i=1;i<m_fullLayers.size();i++)
			m_fullLayers[i].getGrad(m_fullLayers[i-1].m_fullData);
		m_fullLayers[0].getGrad(m_convLayers[m_convLayers.size()-1].m_poolData);
	}
	else
		m_sfm.getGrad(m_convLayers[m_convLayers.size()-1].m_poolData);

	for(int i=1;i<m_convLayers.size();i++)
		m_convLayers[i].getGrad(m_convLayers[i-1].m_poolData);
	m_convLayers[0].getGrad(srcData);

	return PK_SUCCESS;
}

int CCNN::updateWeights()
{
	for(int i=0;i<m_convLayers.size();i++)
		m_convLayers[i].updateWeight(m_sgd.momentum,m_sgd.alpha);

	for(int i=0;i<m_fullLayers.size();i++)
		m_fullLayers[i].updateWeight(m_sgd.momentum,m_sgd.alpha);
	m_sfm.updateWeight(m_sgd.momentum,m_sgd.alpha);
	return PK_SUCCESS;
}

int CCNN::save(const char* path)
{
	FILE* fp=NULL;
	fp=fopen(path,"wb");
	if(fp==NULL)
		return PK_FAIL;
	for(int i=0;i<m_convLayers.size();i++)
		m_convLayers[i].save(fp);
	for(int i=0;i<m_fullLayers.size();i++)
		m_fullLayers[i].save(fp);
	m_sfm.save(fp);
	fclose(fp);
	return PK_SUCCESS;
}

int CCNN::load(const char* path)
{
	FILE* fp=NULL;
	fp=fopen(path,"rb");
	if(fp==NULL)
		return PK_NOT_FILE;
	for(int i=0;i<m_convLayers.size();i++)
		m_convLayers[i].load(fp);
	for(int i=0;i<m_fullLayers.size();i++)
		m_fullLayers[i].load(fp);
	m_sfm.load(fp);
	fclose(fp);
	return PK_SUCCESS;
}

int CCNN::init(const char*path)
{
	int nRet=0;
	CCNNConfig c;
	nRet=c.loadConfig(path);

	if(c.gradCheck)
	{
		setParam(c.sgd,0.0,NL_SOFT_PLUS);
		nRet=init(c.convs,c.fulls,c.sfm,c.inputDim,c.inputChannel,2);
	}
	else
	{
		setParam(c.sgd,0.5,NL_RELU);
		nRet=init(c.convs,c.fulls,c.sfm,c.inputDim,c.inputChannel,c.sgd.minibatch);
	}
	return nRet;
}