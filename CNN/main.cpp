#include <Windows.h>
#include<iostream>
#include "../net/solve.h"

/*int cutImage(CpkMat&images,CpkMat&src,vector<pk::Rect>&rects,int destSize,float scaleFactor=1.1,int minFaceSize=20)
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
		tmp.setData(dd.RowVector(),i,i+1,0,tmp.Col);
	}
	images=tmp;
	return PK_SUCCESS;
}*/

int main()
{
	CNET net;
	TCHAR exePath[256]={0};
	GetModuleFileName(NULL,exePath, MAX_PATH); 
	(strrchr(exePath,'\\'))[1] = 0;

	TCHAR dataPath[256]={0};

	sprintf(dataPath,"%sdata\\config",exePath);
	std::vector<std::string> configs;
	CTools::findDirectsOrFiles(dataPath,configs,"config;");
	std::cout<<"选择配置文件:"<<std::endl;
	for(int i=0;i<configs.size();i++)
		std::cout<< i << ":" << configs[i] <<std::endl;
	int choiceDataSet=-1;
	for(;;)
	{
		std::cout<<"选择:";
		std::cin>>choiceDataSet;
		if(choiceDataSet>=0&&choiceDataSet<configs.size())
			break;
	}

	CSolve solve;
	solve.init(configs[choiceDataSet].c_str());
	solve.run();
	system("pause");
	return 0;
 }