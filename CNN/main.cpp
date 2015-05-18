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

//			imgs.push_back(tmpV);
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


#include <Windows.h>

void findDirectory(const char*directoryPath,std::vector<std::string>& lists)
{
#if defined(_MSC_VER)
	char path[256]={0};
	strcpy(path,directoryPath);
	strcat(path,"\\*.*");

	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(path, &fd);
	if(hFind != INVALID_HANDLE_VALUE)
	{
		do
		{
			if( lstrcmp(fd.cFileName, TEXT(".")) == 0 ||
				lstrcmp(fd.cFileName, TEXT("..")) == 0 )
			{
				continue;
			}
			strcpy(path,directoryPath);
			strcat(path,"\\");
			strcat(path,fd.cFileName);
			lists.push_back(path);
		}
		while(::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
#endif
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

	CpkMat trainImgs,testImgs,predImgs;
	std::vector<int> trainLabels,testLabels,predLabels;
	std::vector<std::string> corrpath,errorPath;
	TCHAR dataPath[256]={0};
	TCHAR labelsPath[256]={0};

	std::cout<<"选择数据集:"<<std::endl
			 <<"1:MNIST"<<std::endl
			 <<"2:CIFAR10"<<std::endl
			 <<"3:mySelf"<<std::endl;
	int choiceDataSet=-1;
	for(;;)
	{
		std::cout<<"选择:";
		std::cin>>choiceDataSet;
		if(choiceDataSet>0&&choiceDataSet<4)
			break;
	}

	std::cout<<std::endl
			 <<"选择操作:"<<std::endl
			 <<"0:梯度测试"<<std::endl
			 <<"1:训练"<<std::endl
			 <<"2:测试"<<std::endl
			 <<"3:预测"<<std::endl;
	int runMode=-1;
	for(;;)
	{
		std::cout<<"选择:";
		std::cin>>runMode;
		if(runMode<4&&runMode>=0)
			break;
	}

	std::cout<<std::endl
			 <<"数据预处理操作:"<<std::endl
			 <<"0:不用"<<std::endl
			 <<"1:ZCA白化"<<std::endl;
	int imgPreDeal=0;
	for(;;)
	{
		std::cout<<"选择:";
		std::cin>>imgPreDeal;
		if(imgPreDeal>=0&&imgPreDeal<2)
			break;
	}

	if(choiceDataSet==1)
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
	}
	else if(choiceDataSet==2)
	{
		sprintf(dataPath,_T("%sdata\\CIFAR10"),path);
//			if(LoadCIFAR10(dataPath,5,trainImgs,trainLabels)!=PK_SUCCESS)
//			{
//				printf("加载训练集失败!");
//				return 0;
//			}
//			if(LoadCIFAR10(dataPath,1,testImgs,testLabels)!=PK_SUCCESS)
//			{
//				printf("加载测试集失败!");
//				return 0;
//			}
			sprintf(dataPath,_T("%sdata\\config\\cnn_cifar10.config"),path);
	}
	else if(choiceDataSet==3)
	{
		CpkMat d;
		int index=0;
		if(runMode==0||runMode==1)
		{
			findDirectory("D:\\CUDA\\project\\CNN\\Release\\data\\faces\\faces",corrpath);
			findDirectory("D:\\CUDA\\project\\CNN\\Release\\data\\faces\\nofaces",errorPath);
			trainImgs.Resize(15000*2,24*24,1,CpkMat::DATA_BYTE);

			
			for(int i=0;i<15000;i++)
			{
				pk::imread(corrpath[i].c_str(),d);
				pk::zoom(d,24,24,d,pk::BILINEAR);
				trainImgs.setRowData(index++,d.RowVector());
				trainLabels.push_back(1);

				pk::imread(errorPath[i].c_str(),d);
				pk::zoom(d,24,24,d,pk::BILINEAR);
				trainImgs.setRowData(index++,d.RowVector());
				trainLabels.push_back(0);
			}
			trainImgs.Resize(CpkMat::DATA_DOUBLE);
			trainImgs=trainImgs/255;

			corrpath.clear();
			errorPath.clear();
		}
		if(runMode==2)
		{
			findDirectory("D:\\CUDA\\project\\CNN\\Release\\data\\faces\\facesTest",corrpath);
			findDirectory("D:\\CUDA\\project\\CNN\\Release\\data\\faces\\nofacesTest",errorPath);

			testImgs.Resize(5000*2,24*24,1,CpkMat::DATA_BYTE);
			for(int i=0;i<5000;i++)
			{
				pk::imread(corrpath[i].c_str(),d);
				pk::zoom(d,24,24,d,pk::BILINEAR);
				testImgs.setRowData(index++,d.RowVector());
				testLabels.push_back(1);

				pk::imread(errorPath[i].c_str(),d);
				pk::zoom(d,24,24,d,pk::BILINEAR);
				testImgs.setRowData(index++,d.RowVector());
				testLabels.push_back(0);
			}
			testImgs.Resize(CpkMat::DATA_DOUBLE);
			testImgs=testImgs/255;

			corrpath.clear();
			errorPath.clear();
		}
		else
		{
			findDirectory("D:\\CUDA\\project\\CNN\\Release\\data\\faces\\pred",corrpath);
			index=0;
			predImgs.Resize(3700,24*24,1,CpkMat::DATA_BYTE);
			for(int i=0;i<3700;i++)
			{
				pk::imread(corrpath[i].c_str(),d);
				pk::zoom(d,24,24,d,pk::BILINEAR);
				predImgs.setRowData(index++,d.RowVector());
			}
			predImgs.Resize(CpkMat::DATA_DOUBLE);
			predImgs=predImgs/255;
		}
		sprintf(dataPath,_T("%sdata\\config\\cnn_faces.config"),path);
	}

	g_cnn.init(dataPath);

	TCHAR exe_path[256]={0};
	GetModuleFileName(NULL,exe_path, MAX_PATH); 
	(_tcsrchr(exe_path, _T('\\')))[1] = 0;

	if(runMode==0)
	{
		double r=r=g_cnn.computeNumericalGradient(trainImgs,trainLabels);
		printf("梯度误差:%e",r);
		if(r>1e-9)
			printf("(算法有问题)\n");
		else
			printf("(检测通过)\n");
	}
	else if(runMode==1)
	{
		if(imgPreDeal==1)
		{
			pk::CPCA pca;
			trainImgs=trainImgs.Transpose();

			CpkMat tmpMat;
			pk::avg(tmpMat,trainImgs,pk::DATA_COLS);
			trainImgs=pk::subVec(trainImgs,tmpMat,pk::DATA_ROWS);

			pca.run(trainImgs,pk::DATA_COLS,pk::CPCA::ZCA_WHITE);
			sprintf(dataPath,_T("%sdata\\pca.dat"),exe_path);
			pca.saveData(dataPath);

			pca.project(trainImgs,pk::DATA_COLS);
			trainImgs=trainImgs.Transpose();
		}
		g_cnn.cnnTrain(trainImgs,trainLabels);

		sprintf(dataPath,_T("%sdata\\theta.dat"),exe_path);
		g_cnn.save(dataPath);
	}
	else if(runMode==2)
	{
		sprintf(dataPath,_T("%sdata\\theta.dat"),exe_path);
		if(g_cnn.load(dataPath)!=PK_SUCCESS)
		{
			printf("找不到权值数据所在文件或文件有问题，请重新生成!");
			return PK_FAIL;
		}

		if(imgPreDeal==1)
		{
			pk::CPCA pca;
			testImgs=testImgs.Transpose();

			CpkMat tmpMat;
			pk::avg(tmpMat,testImgs,pk::DATA_COLS);
			testImgs=pk::subVec(testImgs,tmpMat,pk::DATA_ROWS);

			sprintf(dataPath,_T("%sdata\\pca.dat"),exe_path);
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
			testImgs.GetData(tmpImg,i,i+testBatch,0,testImgs.lineSize*testImgs.Depth);
			g_cnn.cnnRun(pred,tmpImg);
			printf("测试进度:%.2lf\b\b\b\b\b\b\b\b\b\b\b\b\b\b",100.0*(i+testBatch)/testImgs.Row);
		}
		for(int i=0;i<testLabels.size();i++)
		{
			if(pred[i]==testLabels[i])
				right++;
		}
		printf("\n%.2lf\n",right/(double)testLabels.size()*100);
	}
	else if(runMode==3)
	{
		sprintf(dataPath,_T("%sdata\\theta.dat"),exe_path);
		if(g_cnn.load(dataPath)!=PK_SUCCESS)
		{
			printf("找不到权值数据所在文件或文件有问题，请重新生成!");
			return PK_FAIL;
		}

		if(imgPreDeal==1)
		{
			pk::CPCA pca;
			predImgs=predImgs.Transpose();

			CpkMat tmpMat;
			pk::avg(tmpMat,predImgs,pk::DATA_COLS);
			predImgs=pk::subVec(predImgs,tmpMat,pk::DATA_ROWS);

			sprintf(dataPath,_T("%sdata\\pca.dat"),exe_path);
			pca.loadData(dataPath);

			pca.project(predImgs,pk::DATA_COLS);
			predImgs=predImgs.Transpose();
		}
		
		std::vector<double> pred;
		Data tmpImg;
		int testBatch=g_cnn.getMinSGDInfo().minibatch;
		for(int i=0;i<predImgs.Row;i+=testBatch)
		{
			predImgs.GetData(tmpImg,i,i+testBatch,0,predImgs.lineSize*predImgs.Depth);
			g_cnn.cnnRun(pred,tmpImg);
			printf("测试进度:%.2lf\b\b\b\b\b\b\b\b\b\b\b\b\b\b",100.0*(i+testBatch)/predImgs.Row);
		}
		for(int i=0;i<pred.size();i++)
			predLabels.push_back(pred[i]);
	}

	for(int i=0;i<predLabels.size();i++)
	{
		if(predLabels[i]==1)
 		{
			std::string mess="D:\\CUDA\\project\\CNN\\Release\\data\\faces\\real\\";
 			int indexx=corrpath[i].rfind("\\")+1;
 			mess+=corrpath[i].substr(indexx,corrpath[i].length()-indexx);
//			pk::imwrite(mess.c_str(),predImgs[i][0]);
			CopyFile(corrpath[i].c_str(),mess.c_str(),FALSE);
		}
 	}
	system("pause");

	

	return 0;
 }