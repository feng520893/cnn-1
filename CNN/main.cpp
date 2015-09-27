#include <tchar.h>
#include <Windows.h>
#include<vector>

#include "Cnn.h"
#include "..\Common\imgProc\PKImageFunctions.h"
#include "..\Common\opencv\transplant.h"
#include"..\CNNConfig\CNNConfig.h"

using namespace std;

int readFlippedInteger(FILE *fp)
{   
	int ret = 0;    
	BYTE *temp;   
	temp = (BYTE*)(&ret);  
	fread(&temp[3], sizeof(BYTE), 1, fp);   
	fread(&temp[2], sizeof(BYTE), 1, fp);   
	fread(&temp[1], sizeof(BYTE), 1, fp); 
	fread(&temp[0], sizeof(BYTE), 1, fp); 
	return ret; 
}

int LoadMNIST(const char* imagesPath,CpkMat& imgs,const char* labelsPath, std::vector<int>& labels)
{
	FILE *fp=fopen(imagesPath,"rb+");
	if(fp==NULL)
		return PK_NOT_FILE;
	int m=readFlippedInteger(fp);
	int number=readFlippedInteger(fp);
	int row=readFlippedInteger(fp);
	int col=readFlippedInteger(fp);
	
	imgs.Resize(number,col*row,1,CpkMat::DATA_BYTE);
	BYTE* buffer=imgs.GetData<BYTE>();
	fread(buffer,1,row*col*number,fp);
	fclose(fp);
	imgs.Resize(CpkMat::DATA_DOUBLE);
	imgs=imgs/255;

	FILE* fp2=fopen(labelsPath,"rb+");
	if(fp2==NULL)
		return PK_NOT_FILE;
	m=readFlippedInteger(fp2);
	number=readFlippedInteger(fp2);

	
	for(int j=0;j<number;j++)
	{
		char label;
		fread(&label,1,sizeof(char),fp2);
		int dd=label;
		labels.push_back(dd);
	}

	fclose(fp2);
	return PK_SUCCESS;
}

int LoadCIFAR10(const char* imagesPath,int number,Data& imgs, std::vector<int>& labels)
{
	char dataPath[250]={0};
	BYTE buffer[32*32]={0};
	imgs.Resize(number*10000,32*32,3,CpkMat::DATA_DOUBLE);
	for(int i=0;i<number;i++)
	{
		if(number==1)
			sprintf(dataPath,_T("%s\\test_batch.bin"),imagesPath);
		else
			sprintf(dataPath,_T("%s\\data_batch_%d.bin"),imagesPath,i+1);
		FILE *fp=fopen(dataPath,"rb+");
		if(fp==NULL)
		{
			printf("ѵ����������\n");
			return -1;
		}	
		for(int j=0;j<10000;j++)
		{
			char label;
			fread(&label,1,sizeof(char),fp);
			int dd=label;
			labels.push_back(dd);
			fread(buffer,1,32*32,fp);
			CpkMat d(32,32,1,CpkMat::DATA_BYTE,buffer);
//			pk::imwrite("d:\\1.bmp",d);
			d.Resize(CpkMat::DATA_DOUBLE);
			d=d/255;
			d=d.RowVector();
			memcpy(imgs.GetData<double>()+(i*10000+j)*32*32*3,d.GetData<double>(),32*32*sizeof(double));

			fread(buffer,1,32*32,fp);
			d.Resize(32,32,1,CpkMat::DATA_BYTE,buffer);
//			pk::imwrite("d:\\2.bmp",d);
			d.Resize(CpkMat::DATA_DOUBLE);
			d=d/255;
			d=d.RowVector();
			memcpy(imgs.GetData<double>()+(i*10000+j)*32*32*3+32*32,d.GetData<double>(),32*32*sizeof(double));

			fread(buffer,1,32*32,fp);
			d.Resize(32,32,1,CpkMat::DATA_BYTE,buffer);
//			pk::imwrite("d:\\3.bmp",d);
			d.Resize(CpkMat::DATA_DOUBLE);
			d=d/255;
			d=d.RowVector();
			memcpy(imgs.GetData<double>()+(i*10000+j)*32*32*3+32*32*2,d.GetData<double>(),32*32*sizeof(double));
		}
	fclose(fp);
	}
	return PK_SUCCESS;
}

int cutImage(CpkMat&images,CpkMat&src,vector<pk::Rect>&rects,int destSize,float scaleFactor=1.1,int minFaceSize=20)
{
	int index=0;
	while(minFaceSize<(src.Row/2+1)||minFaceSize>(src.Col/2+1))
	{
		for(int i=0;i<src.Row-minFaceSize;i+=minFaceSize)
			for(int j=0;j<src.Col-minFaceSize;j+=minFaceSize)
				rects.push_back(pk::Rect(j,i,minFaceSize,minFaceSize));
		minFaceSize*=scaleFactor;
	}

	CpkMat tmp(rects.size(),destSize*destSize,src.Depth,CpkMat::DATA_DOUBLE);
	for(int i=0;i<rects.size();i++)
	{
		CpkMat dd;
		src.GetData(dd,rects[i].y,rects[i].y+rects[i].height,rects[i].x,rects[i].x+rects[i].width);
		pk::zoom(dd,destSize,destSize,dd,pk::BILINEAR);
		dd.Resize(CpkMat::DATA_DOUBLE);
		dd=dd/255;
		tmp.setRowData(i,dd.RowVector());
	}
	images=tmp;
	return PK_SUCCESS;
}

int main()
{
	CCNN g_cnn;
	TCHAR exePath[256]={0};
	GetModuleFileName(NULL,exePath, MAX_PATH); 
	(_tcsrchr(exePath, _T('\\')))[1] = 0;

	CpkMat trainImgs,testImgs,predImgs;
	std::vector<int> trainLabels,testLabels;
	std::vector<std::string> corrpath,errorPath;
	TCHAR dataPath[256]={0};
	TCHAR labelsPath[256]={0};

	std::cout<<"ѡ�����ݼ�:"<<std::endl
			 <<"1:MNIST"<<std::endl
			 <<"2:CIFAR10"<<std::endl
			 <<"3:mySelf"<<std::endl;
	int choiceDataSet=-1;
	for(;;)
	{
		std::cout<<"ѡ��:";
		std::cin>>choiceDataSet;
		if(choiceDataSet>0&&choiceDataSet<4)
			break;
	}

	std::cout<<std::endl
			 <<"ѡ�����:"<<std::endl
//			 <<"0:�ݶȲ���"<<std::endl
			 <<"1:ѵ��"<<std::endl
			 <<"2:����ѵ��"<<std::endl
			 <<"3:����(ֻ֧��MNIST,CIFAR10)"<<std::endl
			 <<"4:Ԥ��(ֻ֧��MYSELFT)"<<std::endl;
	int runMode=-1;
	for(;;)
	{
		std::cout<<"ѡ��:";
		std::cin>>runMode;
		if(runMode<5&&runMode>=0)
			break;
	}

	int imgPreDeal=0;
/*	std::cout<<std::endl
			 <<"����Ԥ�������:"<<std::endl
			 <<"0:����"<<std::endl
			 <<"1:ZCA�׻�"<<std::endl;
	for(;;)
	{
		std::cout<<"ѡ��:";
		std::cin>>imgPreDeal;
		if(imgPreDeal>=0&&imgPreDeal<2)
			break;
	}*/

	switch(choiceDataSet)
	{
	case 1:
		sprintf(dataPath,_T("%sdata\\config\\cnn_mnist.config"),exePath);
		break;
	case 2:
		sprintf(dataPath,_T("%sdata\\config\\cnn_cifar10.config"),exePath);
		break;
	default:
		sprintf(dataPath,_T("%sdata\\config\\cnn_myselft.config"),exePath);
		break;
	}
	CCNNConfig c;
	int nRet=c.loadConfig(dataPath);
	if(nRet!=PK_SUCCESS)
	{
		printf("���������ļ�ʧ��!\n\n");
		goto error;
	}
	if(c.dataSetsPath.empty())
	{
		printf("����д���ݼ������ļ���·����\n\n");
		goto error;
	}
	if(choiceDataSet==1)
	{
		sprintf(dataPath,_T("%s\\train-images-idx3-ubyte"),c.dataSetsPath.c_str());
		sprintf(labelsPath,_T("%s\\train-labels-idx1-ubyte"),c.dataSetsPath.c_str());
		if(LoadMNIST(dataPath,trainImgs,labelsPath,trainLabels)!=PK_SUCCESS)
		{
			char mess[1024]={0};
			sprintf(mess,"%s��%s�����ڶ�Ӧ�ļ�",dataPath,labelsPath);
			printf(mess);
			goto error;
		}
		sprintf(dataPath,_T("%s\\t10k-images-idx3-ubyte"),c.dataSetsPath.c_str());
		sprintf(labelsPath,_T("%s\\t10k-labels-idx1-ubyte"),c.dataSetsPath.c_str());
		if(LoadMNIST(dataPath,testImgs,labelsPath,testLabels)!=PK_SUCCESS)
		{
			char mess[1024]={0};
			sprintf(mess,"%s��%s�����ڶ�Ӧ�ļ�",dataPath,labelsPath);
			printf(mess);
			goto error;
		}
	}
	else if(choiceDataSet==2)
	{
		if(LoadCIFAR10(c.dataSetsPath.c_str(),4,trainImgs,trainLabels)!=PK_SUCCESS)
		{
			char mess[1024]={0};
			sprintf(mess,"%s�����ڶ�Ӧ�ļ�",dataPath,c.dataSetsPath.c_str());
			printf(mess);
			goto error;
		}
		if(LoadCIFAR10(dataPath,1,testImgs,testLabels)!=PK_SUCCESS)
		{
			char mess[1024]={0};
			sprintf(mess,"%s�����ڶ�Ӧ�ļ�",dataPath,c.dataSetsPath.c_str());
			printf(mess);
			goto error;
		}
	}
	else if(choiceDataSet==3)
	{
		CpkMat d;
		int index=0;
		if(runMode==1||runMode==2)
		{
			int total=0;
			std::vector<std::string> directs;
			pk::findDirectsOrFiles(c.dataSetsPath.c_str(),directs,true);
			if(directs.empty())
			{
				printf("ѵ����Ϊ��!");
				goto error;
			}
			trainImgs.Resize(25000*2,c.imgDim*c.imgDim,1,CpkMat::DATA_DOUBLE);
			for(int i=0;i<directs.size()&&total<50000;i++)
			{
				std::vector<std::string> files;
				pk::findDirectsOrFiles(directs[i].c_str(),files);
				for(int j=0;j<files.size()&&total<50000;j++,total++)
				{
					pk::imread(files[j].c_str(),d);
					pk::ChangeImageFormat(d,d,pk::BGR2GRAY);
					pk::zoomMidImage(d,c.imgDim,pk::BILINEAR);
	
					d.Resize(CpkMat::DATA_DOUBLE);
					d=d/255;

					trainImgs.setRowData(index++,d.RowVector());
					trainLabels.push_back(i);
				}
			}
			if(total<50000)
				trainImgs.Row=total;
		}
	}
	g_cnn.setParam(c.sgd,c.activeFunType,c.runMode);
	nRet=g_cnn.init(c.cnnLayers);

	if(runMode==0)
	{
		double r=r=g_cnn.computeNumericalGradient(trainImgs,trainLabels);
		printf("�ݶ����:%e",r);
		if(r>1e-9)
			printf("(�㷨������)\n");
		else
			printf("(���ͨ��)\n");
	}
	else if(runMode==1||runMode==2)
	{
		if(imgPreDeal==1)
		{
			pk::CPCA pca;
			trainImgs=trainImgs.Transpose();

			CpkMat tmpMat;
			pk::avg(tmpMat,trainImgs,pk::DATA_COLS);
			trainImgs=pk::subVec(trainImgs,tmpMat,pk::DATA_ROWS);

			pca.run(trainImgs,pk::DATA_COLS,pk::CPCA::ZCA_WHITE);
			sprintf(dataPath,_T("%sdata\\pca.dat"),exePath);
			pca.saveData(dataPath);

			pca.project(trainImgs,pk::DATA_COLS);
			trainImgs=trainImgs.Transpose();
		}

		sprintf(dataPath,_T("%sdata\\theta.dat"),exePath);

		if(runMode==2)
		{
			if(g_cnn.load(dataPath)!=PK_SUCCESS)
			{
				printf("�Ҳ���Ȩֵ���������ļ����ļ������⣬����������!");
				goto error;
			}
		}
		
		g_cnn.cnnTrain(trainImgs,trainLabels,dataPath,c.maxSaveCount);

		g_cnn.save(dataPath);
	}
	else if(runMode==3)
	{
		sprintf(dataPath,_T("%sdata\\theta.dat"),exePath);
		if(g_cnn.load(dataPath)!=PK_SUCCESS)
		{
			printf("�Ҳ���Ȩֵ���������ļ����ļ������⣬����������!");
			goto error;
		}

		if(imgPreDeal==1)
		{
			pk::CPCA pca;
			testImgs=testImgs.Transpose();

			CpkMat tmpMat;
			pk::avg(tmpMat,testImgs,pk::DATA_COLS);
			testImgs=pk::subVec(testImgs,tmpMat,pk::DATA_ROWS);

			sprintf(dataPath,_T("%sdata\\pca.dat"),exePath);
			pca.loadData(dataPath);

			pca.project(testImgs,pk::DATA_COLS);
			testImgs=testImgs.Transpose();
		}

		std::vector<double> pred;
		int right=0;
		Data tmpImg;
		int testBatch=g_cnn.getMinSGDInfo().minibatch;
		for(int i=0;i<testImgs.Row;i+=testBatch)
		{
			testImgs.GetData(tmpImg,i,i+testBatch,0,testImgs.lineSize);
			g_cnn.cnnRun(pred,tmpImg);
			printf("���Խ���:%.2lf\b\b\b\b\b\b\b\b\b\b\b\b\b\b",100.0*(i+testBatch)/testImgs.Row);
		}
		for(int i=0;i<testLabels.size();i++)
		{
			if(pred[i]==testLabels[i])
				right++;
		}
		printf("\n%.2lf\n",right/(double)testLabels.size()*100);
	}
	else if(runMode==4)
	{
		sprintf(dataPath,_T("%sdata\\theta.dat"),exePath);
		if(g_cnn.load(dataPath)!=PK_SUCCESS)
		{
			printf("�Ҳ���Ȩֵ���������ļ����ļ������⣬����������!");
			goto error;
		}

		int total=0;
		std::vector<std::string> directs;
		pk::findDirectsOrFiles(c.dataSetsPath.c_str(),directs,true);
		if(directs.empty())
		{
			printf("���Լ�Ϊ��!");
			goto error;
		}
		predImgs.Resize(10000,c.imgDim*c.imgDim,1,CpkMat::DATA_DOUBLE);

		if(imgPreDeal==1)
		{
			pk::CPCA pca;
			predImgs=predImgs.Transpose();

			CpkMat tmpMat;
			pk::avg(tmpMat,predImgs,pk::DATA_COLS);
			predImgs=pk::subVec(predImgs,tmpMat,pk::DATA_ROWS);

			sprintf(dataPath,_T("%sdata\\pca.dat"),exePath);
			pca.loadData(dataPath);

			pca.project(predImgs,pk::DATA_COLS);
			predImgs=predImgs.Transpose();
		}
		
		std::vector<double> pred;
		Data tmpImg;
		int testBatch=g_cnn.getMinSGDInfo().minibatch;
		int index=0;
		for(int i=0;i<directs.size();i++)
		{
			std::vector<std::string> files;
			pk::findDirectsOrFiles(directs[i].c_str(),files);
			for(int j=0;j<files.size();j++)
			{
				CpkMat d;
				pk::imread(files[j].c_str(),d);
				pk::ChangeImageFormat(d,d,pk::BGR2GRAY);
				pk::zoomMidImage(d,c.imgDim,pk::BILINEAR);
	
				d.Resize(CpkMat::DATA_DOUBLE);
				d=d/255;

				predImgs.setRowData(j,d.RowVector());
			}
			for(int k=0;k<files.size();k+=testBatch)
			{
				predImgs.GetData(tmpImg,k,k+testBatch,0,predImgs.lineSize*predImgs.Depth);
				g_cnn.cnnRun(pred,tmpImg);
				printf("��%d���ļ���Ԥ�����:%.2lf\b\b\b\b\b\b\b\b\b\b\b\b\b\b",i+1,100.0*(k+testBatch)/files.size());
			}

			printf("\n");

			int first=directs[i].rfind('\\')+1;
			std::string directName=directs[i].substr(first);

			std::string savePath=std::string(exePath)+std::string("preds\\")+directName;
			ofstream out(savePath+std::string(".txt"));
			for(int i=0;i<files.size();i++)
				out << files[i] << "  ���: " << pred[i] << std::endl;
			out.close();
		}
	}
	system("pause");
	return 0;
error:
	system("pause");
	return -1;
 }