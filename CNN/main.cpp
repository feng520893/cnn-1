#include <tchar.h>
#include "Cnn.h"
#include "..\\Common\\PKImageFunctions.h"

CCNN g_cnn;

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

int LoadMNIST(const char* imagesPath,Data& imgs,const char* labelsPath, std::vector<int>& labels)
{
	FILE *fp=fopen(imagesPath,"rb+");
	if(fp==NULL)
		return PK_NOT_FILE;
	int m=readFlippedInteger(fp);
	int number=readFlippedInteger(fp);
	int row=readFlippedInteger(fp);
	int col=readFlippedInteger(fp);

	FILE* fp2=fopen(labelsPath,"rb+");
	if(fp2==NULL)
		return PK_NOT_FILE;
	m=readFlippedInteger(fp2);
	number=readFlippedInteger(fp2);

	BYTE* buffer=new BYTE[row*col+1];
	for(int j=0;j<number;j++)
	{
		fread(buffer,1,row*col,fp);
		CpkMat d(row,col,1,CpkMat::DATA_BYTE,buffer);
		d.Resize(CpkMat::DATA_DOUBLE);
		d=d/255;
		std::vector<CpkMat> img;
		img.push_back(d);
		imgs.push_back(img);

		char label;
		fread(&label,1,sizeof(char),fp2);
		int dd=label;
		labels.push_back(dd);
	}
	delete [] buffer;
	fclose(fp);
	fclose(fp2);
	return PK_SUCCESS;
}

int LoadCIFAR10(const char* imagesPath,int number,Data& imgs, std::vector<int>& labels)
{
	char dataPath[250]={0};
	BYTE buffer[32*32]={0};
	for(int i=0;i<number;i++)
	{
		if(number==1)
			sprintf(dataPath,_T("%s\\test_batch.bin"),imagesPath);
		else
			sprintf(dataPath,_T("%s\\data_batch_%d.bin"),imagesPath,i+1);
		FILE *fp=fopen(dataPath,"rb+");
		if(fp==NULL)
		{
			printf("训练集不存在\n");
			return -1;
		}	
		for(int j=0;j<10000;j++)
		{
			std::vector<CpkMat> tmpV;
			char label;
			fread(&label,1,sizeof(char),fp);
			int dd=label;
			labels.push_back(dd);
			fread(buffer,1,32*32,fp);
			CpkMat d(32,32,1,CpkMat::DATA_BYTE,buffer);
//			pk::SaveImage("d:\\1.bmp",d);
			d.Resize(CpkMat::DATA_DOUBLE);
			d=d/255;
			tmpV.push_back(d);

			fread(buffer,1,32*32,fp);
			d.Resize(32,32,1,CpkMat::DATA_BYTE,buffer);
			//				pk::SaveImage("d:\\2.bmp",d);
			d.Resize(CpkMat::DATA_DOUBLE);
			d=d/255;
			tmpV.push_back(d);

			fread(buffer,1,32*32,fp);
			d.Resize(32,32,1,CpkMat::DATA_BYTE,buffer);
//			pk::SaveImage("d:\\3.bmp",d);
			d.Resize(CpkMat::DATA_DOUBLE);
			d=d/255;
			tmpV.push_back(d);

			imgs.push_back(tmpV);
		}
	fclose(fp);
	}
	return PK_SUCCESS;
}
/*
int runCIFAR(bool bCheckGrad=true,bool bTest=false)
{
	TCHAR path[256]={0};
	GetModuleFileName(NULL,path, MAX_PATH); 
	(_tcsrchr(path, _T('\\')))[1] = 0;

	if(bCheckGrad)
	{
		testCIFARNetwork(path);
		return 0;
	}
	CCNN::CNNLayer c1;
	int filterDim=5;
	c1.filterH=c1.filterW=filterDim;
	c1.inputSize=3;
	c1.numFeature=10;
	c1.type='c';

	CCNN::CNNLayer c2;
	c2.filterH=c2.filterW=3;
	c2.inputSize=c1.numFeature;
	c2.numFeature=15;
	c2.type='c';

	CCNN::CNNLayer c3;
	c3.filterH=c3.filterW=filterDim;
	c3.inputSize=c2.numFeature;
	c3.numFeature=50;
	c3.type='c';

	CCNN::CNNLayer f;
	f.numFeature=256;
	f.type='f';

	CCNN::CNNLayer s;
	s.numFeature=10;
	s.type='s';
	std::vector<CCNN::CNNLayer> mess;
	mess.push_back(c1);
	mess.push_back(c2);
	mess.push_back(c3);
	mess.push_back(f);
	mess.push_back(f);
	mess.push_back(s);

	Data imgs;
	std::vector<int> labels;

	TCHAR dataPath[256]={0};

	BYTE* buffer=new BYTE[32*32+1];
	if(!bTest)
	{
		for(int i=0;i<5;i++)
		{
			sprintf(dataPath,_T("%sdata\\CIFAR10\\data_batch_%d.bin"),path,i+1);
			FILE *fp=fopen(dataPath,"rb+");
			if(fp==NULL)
			{
				printf("训练集不存在\n");
				return -1;
			}
			
			for(int j=0;j<10000;j++)
			{
				std::vector<CpkMat> tmpV;
				char label;
				fread(&label,1,sizeof(char),fp);
				int dd=label;
				labels.push_back(dd);
				fread(buffer,1,32*32,fp);
				CpkMat d(32,32,1,CpkMat::DATA_BYTE,buffer);
//				pk::SaveImage("d:\\1.bmp",d);
				d.Resize(CpkMat::DATA_DOUBLE);
				d=d/255;
				tmpV.push_back(d);

				fread(buffer,1,32*32,fp);
				d.Resize(32,32,1,CpkMat::DATA_BYTE,buffer);
//				pk::SaveImage("d:\\2.bmp",d);
				d.Resize(CpkMat::DATA_DOUBLE);
				d=d/255;
				tmpV.push_back(d);

				fread(buffer,1,32*32,fp);
				d.Resize(32,32,1,CpkMat::DATA_BYTE,buffer);
//				pk::SaveImage("d:\\3.bmp",d);
				d.Resize(CpkMat::DATA_DOUBLE);
				d=d/255;
				tmpV.push_back(d);

				imgs.push_back(tmpV);
			}
			fclose(fp);
		}
	}
	else
	{
		sprintf(dataPath,_T("%sdata\\test_batch.bin"),path);
		FILE *fp=fopen(dataPath,"rb+");

		for(int j=0;j<10000;j++)
		{
			std::vector<CpkMat> tmpV;
			char label;
			fread(&label,1,sizeof(char),fp);
			int dd=label;
			labels.push_back(dd);
			fread(buffer,1,32*32,fp);
			CpkMat d(32,32,1,CpkMat::DATA_BYTE,buffer);
//			pk::SaveImage("d:\\1.bmp",d);
			d.Resize(CpkMat::DATA_DOUBLE);
			d=d/255;
			tmpV.push_back(d);

			fread(buffer,1,32*32,fp);
			d.Resize(32,32,1,CpkMat::DATA_BYTE,buffer);
//			pk::SaveImage("d:\\2.bmp",d);
			d.Resize(CpkMat::DATA_DOUBLE);
			d=d/255;
			tmpV.push_back(d);

			fread(buffer,1,32*32,fp);
			d.Resize(32,32,1,CpkMat::DATA_BYTE,buffer);
//			pk::SaveImage("d:\\3.bmp",d);
			d.Resize(CpkMat::DATA_DOUBLE);
			d=d/255;
			tmpV.push_back(d);

			imgs.push_back(tmpV);
		}
		fclose(fp);
	}

	MinSGD sgd;
	sgd.alpha=0.005;
	sgd.epoches=30;
	sgd.minibatch=100;
	sgd.momentum=0.9;
	g_cnn.SetParam(sgd,0.5);

	if(!bTest)
	{
		g_cnn.init(mess,32,32);
		g_cnn.cnnTrain(imgs,labels);
		sprintf(dataPath,_T("%sdata\\theta.dat"),path);
		g_cnn.save(dataPath);
	}
	else
	{
		sprintf(dataPath,_T("%sdata\\theta.dat"),path);
		g_cnn.init(mess,32,32,dataPath);
		CpkMat pred;
		g_cnn.cnnRun(pred,imgs);
		int right=0;
		for(int i=0;i<labels.size();i++)
		{
			if(pred.GetData<int>()[i]==labels[i])
				right++;
		}
		printf("%.2lf",right/(double)labels.size()*100);
	}
	return 0;
}



int runORL(bool bCheckGrad=true,bool bTest=false)
{
	TCHAR path[256]={0};
	GetModuleFileName(NULL,path, MAX_PATH); 
	(_tcsrchr(path, _T('\\')))[1] = 0;


	CCNN::CNNLayer c1;
	int filterDim=5;
	c1.filterH=c1.filterW=9;
	c1.inputSize=1;
	c1.numFeature=60;
	c1.type='c';

	CCNN::CNNLayer c2;
	c2.filterH=c2.filterW=5;
	c2.inputSize=c1.numFeature;
	c2.numFeature=100;
	c2.type='c';

	CCNN::CNNLayer c3;
	c3.filterH=c3.filterW=3;
	c3.inputSize=c2.numFeature;
	c3.numFeature=180;
	c3.type='c';

	CCNN::CNNLayer c4;
	c4.filterH=c4.filterW=3;
	c4.inputSize=c3.numFeature;
	c4.numFeature=340;
	c4.type='c';

	CCNN::CNNLayer f;
	f.numFeature=1024;
	f.type='f';

	CCNN::CNNLayer f2;
	f2.numFeature=512;
	f2.type='f';

	CCNN::CNNLayer s;
	s.numFeature=2;
	s.type='s';
	std::vector<CCNN::CNNLayer> mess;
	mess.push_back(c1);
	mess.push_back(c2);
	mess.push_back(c3);
	mess.push_back(c4);
	mess.push_back(f);
	mess.push_back(f);
	mess.push_back(f2);
	mess.push_back(s);

	Data imgs;
	std::vector<int> labels;

	TCHAR dataPath[256]={0};

	MinSGD sgd;
	sgd.alpha=0.00005;
	sgd.epoches=700;
	sgd.minibatch=5;
	sgd.momentum=0.9;
	g_cnn.SetParam(sgd,0.5);

	if(!bTest)
	{
		for(int i=1;i<21;i++)
		{
			if(i<10)
				sprintf(dataPath,"G:\\ORL\\orl00%d.bmp",i);
			else if(i<100)
				sprintf(dataPath,"G:\\ORL\\orl0%d.bmp",i);
			else
				sprintf(dataPath,"G:\\ORL\\orl%d.bmp",i);
			CpkMat d;
			pk::loadImage(dataPath,d);
			pk::zoom(d,88,88,d,pk::ZOOM_TYPE::BILINEAR);
			std::vector<CpkMat> tmpV;
			int label;
			if(i>10&&i<21)
				label=1;
			else
				label=0;
			
			d.Resize(CpkMat::DATA_DOUBLE);
			d=d/255;
			tmpV.push_back(d);
			for(int i=0;i<10;i++)
			{
				imgs.push_back(tmpV);
				labels.push_back(label);
			}
		}

		g_cnn.init(mess,88,88);
		g_cnn.cnnTrain(imgs,labels);
		sprintf(dataPath,_T("%sdata\\theta.dat"),path);
		g_cnn.save(dataPath);
	}
	else
	{
		for(int i=201;i<401;i++)
		{
			if(i<10)
				sprintf(dataPath,"G:\\ORL\\orl00%d.bmp",i);
			else if(i<100)
				sprintf(dataPath,"G:\\ORL\\orl0%d.bmp",i);
			else
				sprintf(dataPath,"G:\\ORL\\orl%d.bmp",i);
			CpkMat d;
			pk::loadImage(dataPath,d);
			pk::zoom(d,88,88,d,pk::ZOOM_TYPE::BILINEAR);
			std::vector<CpkMat> tmpV;
			int label;
			if(i>10&&i<21)
				label=1;
			else
				label=0;
			labels.push_back(label);
			d.Resize(CpkMat::DATA_DOUBLE);
			d=d/255;
			tmpV.push_back(d);
			imgs.push_back(tmpV);
		}

		sprintf(dataPath,_T("%sdata\\theta.dat"),path);
		g_cnn.init(mess,88,88,dataPath);
		CpkMat pred;
		g_cnn.cnnRun(pred,imgs);
		int right=0;
		for(int i=0;i<labels.size();i++)
		{
			if(pred.GetData<int>()[i]!=labels[i])
			{
				printf("error:%d: E-%d  R-%d\n",i,pred.GetData<int>()[i],labels[i]);
			}
			else
				right++;
		}
		printf("%.2lf",right/(double)labels.size()*100);
	}
	return 0;
}
*/


bool InitCUDA()
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



int main()
{
	if(!InitCUDA())
	{
		system("pause");
		return 0;
	}
	TCHAR path[256]={0};
	GetModuleFileName(NULL,path, MAX_PATH); 
	(_tcsrchr(path, _T('\\')))[1] = 0;

	Data trainImgs,testImgs;
	std::vector<int> trainLabels,testLabels;
	TCHAR dataPath[256]={0};
	TCHAR labelsPath[256]={0};

	std::cout<<"选择数据集:"<<std::endl
			 <<"1:MNIST"<<std::endl
			 <<"2:CIFAR10"<<std::endl;
	for(;;)
	{
		int choice=-1;
		std::cout<<"选择:";
		std::cin>>choice;
		if(choice==1)
		{
			sprintf(dataPath,_T("%sdata\\mnist\\train-images-idx3-ubyte"),path);
			sprintf(labelsPath,_T("%sdata\\mnist\\train-labels-idx1-ubyte"),path);
			if(LoadMNIST(dataPath,trainImgs,labelsPath,trainLabels)!=PK_SUCCESS)
			{
				printf("加载训练集失败!");
				return 0;
			}
			sprintf(dataPath,_T("%sdata\\mnist\\t10k-images-idx3-ubyte"),path);
			sprintf(labelsPath,_T("%sdata\\mnist\\t10k-labels-idx1-ubyte"),path);
			if(LoadMNIST(dataPath,testImgs,labelsPath,testLabels)!=PK_SUCCESS)
			{
				printf("加载测试集失败!");
				return 0;
			}
			sprintf(dataPath,_T("%sdata\\config\\cnn_mnist.config"),path);
			break;
		}
		else if(choice==2)
		{
			sprintf(dataPath,_T("%sdata\\CIFAR10"),path);
			if(LoadCIFAR10(dataPath,5,trainImgs,trainLabels)!=PK_SUCCESS)
			{
				printf("加载训练集失败!");
				return 0;
			}
			if(LoadCIFAR10(dataPath,1,testImgs,testLabels)!=PK_SUCCESS)
			{
				printf("加载测试集失败!");
				return 0;
			}
			sprintf(dataPath,_T("%sdata\\config\\cnn_cifar10.config"),path);
			break;
		}
		else
			printf("\b无效选择\n");
	}
	g_cnn.loadConfig(dataPath,trainImgs,trainLabels,testImgs,testLabels);
	system("pause");
	return 0;
 }