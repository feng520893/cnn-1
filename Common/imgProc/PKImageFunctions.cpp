#include "PKImageFunctions.h"
#include "Jpeg/Jpeg.h"
#include"../PKMatFunctions.h"
#include<cmath>
#include <string>
#include<map>

#if _MSC_VER
#include"showImgDlg\ShowImgWinDlg.h"
#endif

namespace pk
{

	std::map<std::string,IShowImgBaseDlg*> g_dlgs;

	typedef std::map<std::string,IShowImgBaseDlg*>::iterator mapDlgIt;

	#pragma pack (2) //2字节对齐
		typedef struct tagBITMAPFILEHEADER
		{
			WORD bfType;
			DWORD bfSize;
			WORD bfReserved1;
			WORD bfReserved2;
			DWORD bfOffBits;
		}BITMAPFILEHEADER;
		#pragma pack () //恢复原有对齐

		typedef struct tagBITMAPINFOHEADER{
			long biSize;
			long biWidth;
			long biHeight;
			short biPlanes;
			short biBitCount;
			unsigned long biCompression;
			unsigned long biSizeImage;//位图的大小(其中包含了为了补齐行数是4的倍数而添加的空字节)，以字节为单位（35-38字节）
			long biXPelsPerMeter;
			long biYPelsPerMeter;
			unsigned long biClrUsed;
			unsigned long biClrImportant;
		}BITMAPINFOHEADER;

	int SaveBmp(const char*path,CpkMat& src)
	{
		FILE *fp;
		int err=fopen_s(&fp,path,"wb");
		if(err!=0)
			return PK_NOT_FILE;

		BITMAPFILEHEADER fileHead;
		fileHead.bfType=0x4D42;
		fileHead.bfSize=54+src.Row*src.lineSize;
		fileHead.bfReserved1=0;
		fileHead.bfReserved2=0;
		fileHead.bfOffBits=54;
		if(src.Depth<3)
		{
			fileHead.bfSize+=1024;
			fileHead.bfOffBits+=1024;
		}

		BITMAPINFOHEADER Infohead;

		Infohead.biSize=40;
		Infohead.biWidth=src.Col;
		Infohead.biHeight=src.Row;
		Infohead.biPlanes=1;
		Infohead.biBitCount=src.Depth*8;
		Infohead.biCompression=0;
		Infohead.biSizeImage=src.Row*src.lineSize;
		Infohead.biXPelsPerMeter=0;
		Infohead.biYPelsPerMeter=0;
		Infohead.biClrUsed=0;
		Infohead.biClrImportant=0;


		fwrite(&fileHead,sizeof(BITMAPFILEHEADER),1,fp);

		fwrite(&Infohead,sizeof(BITMAPINFOHEADER),1,fp);
		if(src.Depth<3)
		{
			char rgb[1024]={0};
			for(int i=0;i<256;i++)
				for(int k=0;k<4;k++)
					rgb[i*4+k]=i;
			fwrite(rgb,1,1024,fp);
		}
		fwrite(src.GetData<BYTE>(),1,src.Row*src.lineSize,fp);
		fclose(fp);
		return PK_SUCCESS;
	}

	int saveJpg(const char* path,CpkMat&src)
	{
		int nRet=PK_SUCCESS;
		BYTE* buff=NULL;
		unsigned long outSize=0;
		int size=(src.lineSize*src.Row);
		if(JpegCompression(&buff,outSize,src.Col,src.Row,src.Depth*8,95,src.GetData<BYTE>(),size))
		{
			FILE *fp;
			fp=fopen(path,"wb");
			if(fp==NULL)
				return PK_OPEN_FILE_ERROR;
			fwrite(buff,1,outSize,fp);
			fclose(fp);
		}
		if(buff)
			delete [] buff;
		return nRet;
	}

	int imwrite(const char*path,CpkMat& src)
	{
		if(src.GetType()!=CpkMat::DATA_BYTE)
			return PK_NOT_ALLOW_OPERATOR;
		int nRet=PK_SUCCESS;
		std::string suffix;
		suffix=path;
		int index=suffix.find('.')+1;
		suffix=suffix.substr(index,suffix.length()-index);
		if(suffix=="jpg"||suffix=="JPG"||suffix=="jpeg"||suffix=="JPEG")
			nRet=saveJpg(path,src);
		else if(suffix=="bmp"||suffix=="BMP")
			nRet=SaveBmp(path,src);
		else
			nRet=PK_NOT_SUPPORT_FORMAT;
		return nRet;
	}

	int loadBmp(FILE* fp,CpkMat& dest)
	{

		BITMAPFILEHEADER fileHead;

		BITMAPINFOHEADER Infohead;

		fread(&fileHead,sizeof(BITMAPFILEHEADER),1,fp);

		fread(&Infohead,sizeof(BITMAPINFOHEADER),1,fp);
		char rgb[1024]={0};
		if(Infohead.biBitCount<24)
			fread(rgb,1,1024,fp);
		int lineByte=(Infohead.biWidth*Infohead.biBitCount/8+3)/4*4;
		//写位图数据
		dest.Resize(Infohead.biHeight,Infohead.biWidth,Infohead.biBitCount/8,CpkMat::DATA_BYTE);
		
		BYTE* buff=dest.GetData<BYTE>();
		
		fseek(fp,fileHead.bfOffBits,SEEK_SET);
		fread(buff,1,Infohead.biHeight*lineByte,fp);
		return PK_SUCCESS;
	}

	int loadJpeg(FILE* fp,CpkMat& dest)
	{
		int nRet;
		fseek(fp,0,SEEK_END);
		int size=ftell(fp);
		fseek(fp,0,SEEK_SET);
		unsigned char* fileData=new unsigned char[size];
		if(fileData==NULL)
			return PK_ALLOCATE_MEMORY_FAIL;
		fread(fileData,1,size,fp);

		BYTE* buffer=NULL;
		unsigned long outSize=0;
		int outW=0,outH=0,outBit=0;
		
		if(JpegDecompress(&buffer,outSize,outW,outH,outBit,fileData,size))
			nRet=dest.Resize(outH,outW,outBit,CpkMat::DATA_BYTE,buffer);
		else
			nRet=PK_FAIL;
		if(fileData)
			delete [] fileData;
		if(buffer)
			delete [] buffer;
		return nRet;
	}

	int imread(const char*path,CpkMat& dest)
	{
		FILE *fp;
		fp=fopen(path,"rb+");
		if(fp==NULL)
			return PK_NOT_FILE;
		int nRet=PK_SUCCESS;
		int c1 = getc(fp);
		int c2 = getc(fp);
		fseek(fp,0,SEEK_SET);
		if (c1 == 0xFF && c2 == 0xD8)
			nRet=loadJpeg(fp,dest);
		else if(c1==0x42&&0x4D)
			nRet=loadBmp(fp,dest);
		else
			nRet=PK_NOT_SUPPORT_FORMAT;
		fclose(fp);
		return nRet;
	}

	int ReadImageRect(CpkMat&dest,int destWidth,const char*data,int x,int y,int srcWidth,int height,int bit)
	{
		int lineSrcByte=(destWidth*(bit/8)+3)/4*4;
		int lineByte=(destWidth*(bit/8)+3)/4*4;
		if(x+lineByte>lineSrcByte)
			return PK_NOT_ALLOW_OPERATOR;
		dest.Resize(height,destWidth,bit/8,CpkMat::DATA_BYTE);
		BYTE* pData=dest.GetData<BYTE>();
		for(int i=y;i<height+y;i++)
			for(int j=x;j<lineByte+x;j++)
				pData[(i-y)*lineByte+j-x]=data[i*srcWidth+j];
		return PK_SUCCESS;
	}

	int RGBtoGrayscale(CpkMat src,CpkMat&dest)
	{
		CpkMat tmp;
		BYTE* pSrc=src.GetData<BYTE>();
		
		if(src.Depth==1)
		{
			tmp.Resize(src.Row,src.Col,1,CpkMat::DATA_BYTE);
			BYTE* pDest=tmp.GetData<BYTE>();
			for(int i=0;i<src.Row;i++)
				for(int j=0;j<src.lineSize;j++)
					pDest[i*src.lineSize+j]=pSrc[i*src.lineSize+j];
		}
		else
		{
			int nPixelByte=src.Depth;
			int nLineByteIn=src.lineSize;
			int nLineByteOut=(src.Col+3)/4*4;
			tmp.Resize(src.Row,src.Col,1,CpkMat::DATA_BYTE);
			tmp.Col=src.Col;
			BYTE* pDest=tmp.GetData<BYTE>();
			for(int i=0;i<src.Row;i++)
			{
				for(int j=0;j<src.Col;j++)
				{
					pDest[i*nLineByteOut+j]=0.11*pSrc[i*nLineByteIn+j*nPixelByte+0]
						+0.59*pSrc[i*nLineByteIn+j*nPixelByte+1]
						+0.30*pSrc[i*nLineByteIn+j*nPixelByte+2]
						+0.5;
				}
			}
		}
		dest=tmp;
		return PK_SUCCESS;
	}

	int RGBtoHSV(CpkMat img,CpkMat&dst)
	{
		if(img.Depth<3)
			return PK_NOT_ALLOW_OPERATOR;
		int w=img.Col;  
		int h=img.Row;  
		dst.Resize(h,w,img.Depth,img.GetType());
		BYTE* pData=img.GetData<BYTE>();
		BYTE* pDest=dst.GetData<BYTE>();
		for(int j=0;j<w;++j)  
		{  
			for(int i=0;i<h;++i)  
			{  
				int b=*(pData+i*img.lineSize+j*3); 
				int g=*(pData+i*img.lineSize+j*3+1); 
				int r=*(pData+i*img.lineSize+j*3+2);   
				int maxval=max(b,max(g,r));  
				int minval=min(b,min(g,r));  
				int v=maxval;  
				double diff=maxval-minval;  
				int s=diff*255/(v+DBL_EPSILON);  

				double h=0;  
				diff=60/(diff+DBL_EPSILON);  
				if(v==r)  
				{  
					h=(g-b)*diff;  
				}  
				else if(v==g)  
				{  
					h=(b-r)*diff+120.f;  
				}  
				else  
				{  
					h=(r-g)*diff+240.f;  
				}  
				if( h<0)  
				{  
					h+=360.f;  
				}  
				*(pDest+i*img.lineSize+j*3)=h/2;  
				*(pDest+i*img.lineSize+j*3+1)=s;  
				*(pDest+i*img.lineSize+j*3+2)=v;  
			}  
		}  
		return PK_SUCCESS;
	}

	int Split(CpkMat&src,std::vector<CpkMat>& mats)
	{
		if(src.Depth<3)
			return PK_NOT_ALLOW_OPERATOR;
		int nLineByteOut=(src.Col+3)/4*4;
		CpkMat colorR,colorG,colorB;
		colorR.Resize(src.Row,nLineByteOut,1,src.GetType());
		colorG.Resize(src.Row,nLineByteOut,1,src.GetType());
		colorB.Resize(src.Row,nLineByteOut,1,src.GetType());
		colorR.Col=colorG.Col=colorB.Col=src.Col;
		BYTE* pR=colorR.GetData<BYTE>();
		BYTE* pG=colorG.GetData<BYTE>();
		BYTE* pB=colorB.GetData<BYTE>();

		BYTE* pSrc=src.GetData<BYTE>();
		for(int i=0;i<src.Row;i++)
		{
			for(int j=0;j<src.Col;j++)
			{
				pB[i*nLineByteOut+j]=pSrc[i*src.lineSize+j*src.Depth];
				pG[i*nLineByteOut+j]=pSrc[i*src.lineSize+j*src.Depth+1];
				pR[i*nLineByteOut+j]=pSrc[i*src.lineSize+j*src.Depth+2];
			}
		}
		mats.push_back(colorB);
		mats.push_back(colorG);
		mats.push_back(colorB);
		return PK_SUCCESS;
	}

	int ChangeImageFormat(CpkMat src,CpkMat&dest,CHANGE_IMAGE_FORMAT type)
	{
		int nRet=PK_SUCCESS;
		switch(type)
		{
		case BGR2HSV:
			nRet=RGBtoHSV(src,dest);
			break;
		case BGR2GRAY:
			nRet=RGBtoGrayscale(src,dest);
			break;
		}
		return nRet;
	}

	int RevColor(CpkMat&dest,CpkMat&src)
	{
		if(src.GetType()!=CpkMat::DATA_BYTE)
			return PK_NOT_ALLOW_OPERATOR;
		CpkMat tmp;
		tmp.Resize(src.Row,src.lineSize,src.Depth,src.GetType());
		tmp.Col=src.Col;
		BYTE* pD=tmp.GetData<BYTE>();
		BYTE* pSrc=src.GetData<BYTE>();
		for(int i=0;i<src.Row*src.lineSize;i++)
			pD[i]=255-pSrc[i];
		dest=tmp;
		return PK_SUCCESS;
	}


	/****************图像缩放铺助函数*****************/
	int getSincVec(float *pfVec, float fPos, float fScale,int cubicSize=3)
	{
		int nPos = (int)fPos;
		fPos -= (float)nPos;
		memset(pfVec, 0, sizeof(float) * cubicSize);
		float fSum = 0.f;
		for (int i=0; i<cubicSize; i++)
		{
			float fTmp = (i - 1.f + fPos) * 3.14159265358979323846f;
			if (fTmp > 0.00001 || fTmp < -0.00001)
				pfVec[i] = sinf(fTmp * fScale) / fTmp;
			else
				pfVec[i] = 1.f;
			fSum += pfVec[i];
		}
		fSum = 1.f / fSum;
		for (int i=0; i<cubicSize; i++)
			pfVec[i] *= fSum;
		return nPos;
	}

	void calcSincMat(float *pfMat, float *pfVecY, float *pfVecX,int cubicSize=3)
	{
		for (int i=0; i<cubicSize; i++)
		{
			for (int j=0; j<cubicSize; j++)
			{
				pfMat[i * cubicSize + j] = pfVecY[i] * pfVecX[j];
			}
		}
	}

	/**********************END************************/

	int zoom(CpkMat&dest,int widthOut,int heighOut,CpkMat&src,ZOOM_TYPE type)
	{
		if(src.GetType()!=CpkMat::DATA_BYTE)
			return PK_NOT_ALLOW_OPERATOR;
		float ratioX=(float)src.Col/widthOut;
		float ratioY=(float)src.Row/heighOut;
		int coordinateX,coordinateY;
		int pix=src.Depth;

		int k;

		CpkMat tmp;

		BYTE* pSrc=src.GetData<BYTE>();
		int lineByteIn=(src.Col*src.Depth+3)/4*4;
		int lineByteOut=(widthOut*src.Depth+3)/4*4;
		tmp.Resize(heighOut,lineByteOut/src.Depth,src.Depth,src.GetType());
		BYTE* pDest=tmp.GetData<BYTE>();
		if(type==NEAREST)
		{
			for(int i=0;i<heighOut;i++)
			{
				for(int j=0;j<widthOut;j++)
				{
					coordinateX=(int)(ratioX*j+0.5);
					coordinateY=(int)(ratioY*i+0.5);
					for(k=0;k<pix;k++)
						pDest[i*lineByteOut+j*pix+k]=pSrc[coordinateY*lineByteIn+coordinateX*pix+k];
				}
			}
		}
		else if(type==BILINEAR)
		{
			for (int i = 0; i < heighOut; i++)
			{
				int tH = (int)(ratioY * i);
				int tH1 = min(tH + 1,src.Row - 1);
				float u = (float)(ratioY * i - tH);
				for (int j = 0; j < widthOut; j++)
				{
					int tW = (int)(ratioX * j); 
					int tW1 = min(tW + 1,src.Col - 1);
					float v = (float)(ratioX * j - tW);
					//f(i+u,j+v) = (1-u)(1-v)f(i,j) + (1-u)vf(i,j+1) + u(1-v)f(i+1,j) + uvf(i+1,j+1) 
					for (int k = 0; k < pix; k++)
					{
						pDest[i * lineByteOut + j * pix + k] = 
							(1 - u)*(1 - v) * pSrc[tH * lineByteIn + tW * pix+ k] + 
							(1 - u)*v*pSrc[tH1 * lineByteIn + tW * pix+ k] + 
							u * (1 - v) * pSrc[tH * lineByteIn + tW1 * pix + k] + 
							u * v * pSrc[tH1 * lineByteIn + tW1 * pix + k];                     
					}            
				}
			}
		}
		else
		{
			const int CUBIC_SIZE = 3;
			float pfVecX[CUBIC_SIZE], pfVecY[CUBIC_SIZE], pfMat[CUBIC_SIZE][CUBIC_SIZE];
			for (int i=1; i<heighOut-1; i++)
			{
				int nSrcY = getSincVec(pfVecY, ratioY * i, 1.f / ratioY);
				for (int j = 1; j < widthOut-1; j++)
				{
					int nSrcX = getSincVec(pfVecX, ratioX * j, 1.f / ratioX);
					
					calcSincMat(pfMat[0], pfVecY, pfVecX);
					BYTE *pbyTmpDes = pDest + i * lineByteOut + j * pix;
					BYTE *pbyTmpSrc = pSrc + nSrcY * lineByteIn + nSrcX * pix;
					for (int k = 0; k < pix; k++)
					{
						float fTmp = pbyTmpSrc[0] * pfMat[1][1];
						fTmp += pbyTmpSrc[-lineByteIn] * pfMat[0][1];
						fTmp += pbyTmpSrc[lineByteIn] * pfMat[2][1];
						fTmp += pbyTmpSrc[-pix] * pfMat[1][0];
						fTmp += pbyTmpSrc[-pix-lineByteIn] * pfMat[0][0];
						fTmp += pbyTmpSrc[-pix+lineByteIn] * pfMat[2][0];
						fTmp += pbyTmpSrc[pix] * pfMat[1][2];
						fTmp += pbyTmpSrc[pix-lineByteIn] * pfMat[0][2];
						fTmp += pbyTmpSrc[pix+lineByteIn] * pfMat[2][2];
						if (fTmp < 0.f) 
							fTmp = 0.f;
						if (fTmp > 255.f) 
							fTmp = 255.f;
						*pbyTmpDes++ = (BYTE)fTmp;
						pbyTmpSrc++;
					} 
				}
			}
		}
		dest=tmp;
		return PK_SUCCESS;
	}

	int zoomMidImage(CpkMat& dd,int zoomSize,ZOOM_TYPE type)
	{
		CpkMat tmp;
		float rate=min(dd.Row,dd.Col)/(float)zoomSize;
		if(dd.Row>dd.Col)
		{
			pk::zoom(tmp,zoomSize,dd.Row/rate,dd,type);
			int mid=tmp.Row/2;
			tmp.GetData(dd,mid-zoomSize/2,mid+zoomSize/2,0,zoomSize);
		}
		else
		{
			pk::zoom(tmp,dd.Col/rate,zoomSize,dd,type);
			int mid=tmp.Col/2;
			tmp.GetData(dd,0,zoomSize,mid-zoomSize/2,mid+zoomSize/2);
		}
		return PK_SUCCESS;
	}

	int UpDown(CpkMat&dest,CpkMat&src)
	{
		CpkMat tmp;
		tmp.Resize(src.Row,src.Col,src.Depth,src.GetType());
		for(int i=0;i<src.Row;i++)
		{
			CpkMat tmp2;
			src.GetData(tmp2,src.Row-i-1,src.Row-i,0,src.Col);
			tmp.setData(tmp2,i,i+1,0,src.Col);
		}
		dest=tmp;
		return PK_SUCCESS;
	}

	int LeftRight(CpkMat&dest,CpkMat&src)
	{
		CpkMat tmp;
		tmp.Resize(src.Row,src.Col,src.Depth,src.GetType());
		for(int i=0;i<src.Col;i++)
		{
			CpkMat tmp2;
			src.GetData(tmp2,0,src.Row,src.Col-i-1,src.Col-i);
			tmp.setData(tmp2,0,src.Row,i,i+1);
		}
		dest=tmp;
		return PK_SUCCESS;
	}

	int Binary(CpkMat&src,int threshold)
	{
		if(src.Depth!=1)
			return PK_NOT_ALLOW_OPERATOR;

		int maxT=0;
		BYTE* pSrc=src.GetData<BYTE>();

		if(threshold==0)
		{
			int nHistArray[256]={0};
			float nHistArrayPro[256]={0};
			for(int i=0;i<src.Row*src.lineSize;i++)
				nHistArray[*(pSrc+i)]++;

			for(int i = 0; i < 256; i++)
				nHistArrayPro[i] = (float)nHistArray[i] / (src.lineSize*src.Row);

			float w0, w1, u0tmp, u1tmp, u0, u1, u, deltaTmp, deltaMax = 0;
			for(int i = 0; i < 256; i++)
			{
				w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = 0;
				for(int j = 0; j < 256; j++)
				{
					if(j <= i)
					{
						w0 += nHistArrayPro[j];
						u0tmp += j * nHistArrayPro[j];
					}
					else
					{
						w1 += nHistArrayPro[j];
						u1tmp += j * nHistArrayPro[j];
					}
				}
				if(w0==0)
					w0=1e-5;
				if(w1==0)
					w1=1e-5;
				u0 = u0tmp / w0;
				u1 = u1tmp / w1;
				u = u0tmp + u1tmp;
				deltaTmp = 
					w0 * ::pow((u0 - u), 2) + w1 * ::pow((u1 - u), 2);
				if(deltaTmp > deltaMax)
				{
					deltaMax = deltaTmp;
					maxT = i;
				}
			}
		}
		else
			maxT=threshold;

		for(int i=0;i<src.Row*src.lineSize;i++)
		{
			if(*(pSrc+i)<maxT)
				*(pSrc+i)=0;
			else
				*(pSrc+i)=255;
		}
		return PK_SUCCESS;
	}


	int GaussSkinModel(CpkMat&destImg,CpkMat&srcImg)
	{
		static double cbMean=117.4361;
		static double crMean=156.5599;
		static double Cov00=160.1301;
		static double Cov01=12.1430;
		static double Cov10=12.1430;
		static double Cov11=299.4574;
		if(srcImg.GetType()!=CpkMat::DATA_BYTE)
			return PK_NOT_ALLOW_OPERATOR;
		BYTE* dataIn=srcImg.GetData<BYTE>();
		int pix=srcImg.Depth;
		//	if(!srcImg->lightCompensation())
		//		return false;
		int imgW=srcImg.Col;
		int imgH=srcImg.Row;
		int lineByte=(imgW*pix+3)/4*4;
		int lineByteOut=(imgW+3)/4*4;
		CpkMat outImg(imgH,imgW,1,srcImg.GetType());
		BYTE* dataOut=outImg.GetData<BYTE>();
		double **pSkipArr=new double*[imgH];
		for(int i=0;i<imgH;i++)
			pSkipArr[i]=new double[imgW];
		char cRGB[4]={};
		for(int i=0;i<imgH;i++)
		{
			for(int j=0;j<imgW;j++)
			{
				for(int k=0;k<pix;k++)
					cRGB[k]=*(dataIn+i*lineByte+j*pix+k);
				int b=(int)cRGB[0]&255;
				int g=(int)cRGB[1]&255;
				int r=(int)cRGB[2]&255;
				//double cr,cb;
				double cb=128-37.797*r/255-74.203*g/255+112*b/255;
				double cr=128+112*r/255-93.786*g/255-18.214*b/255;
				/*if(y>=200)
				{
				cr=(pow(r-y,2.0)*0.713)*((-5000/91)/pow(y-20,2.0)+7);
				cb=(-pow((b-y),2.0)*0.564)*(125/pow(y-200,2.0)-3);
				}
				else
				{
				cr=(r-y)*0.713;
				cb=(b-y)*0.564;
				}*/
				//计算该点属于皮肤区域的概率
				double tt=(cb-cbMean)*((cb-cbMean)*Cov11-(cr-crMean)*Cov10)+(cr-crMean)*(-(cb-cbMean)*Cov01+(cr-crMean)*Cov00);
				tt=(-0.5*tt)/(Cov00*Cov11-Cov01*Cov10);
				pSkipArr[i][j]=::exp(tt);
			}
		}
		//中值滤波使图像平滑过渡
		MedF(pSkipArr,imgW,imgH,9);
		//统计最大肤色相似度
		double max=0.0;
		for(int i=0;i<imgH;i++)
		{
			for(int j=0;j<imgW;j++)
			{
				if(pSkipArr[i][j]>max)
					max=pSkipArr[i][j];
			}
		}
		//肤色相似度归一化
		for(int i=0;i<imgH;i++)
			for(int j=0;j<imgW;j++)
				pSkipArr[i][j]=pSkipArr[i][j]/max;
		//将肤色变换到0~255
		for(int i=0;i<imgH;i++)
			for(int j=0;j<imgW;j++)
				*(dataOut+i*lineByteOut+j)=(int)(pSkipArr[i][j]*255);

		for(int i=0;i<imgH;i++)
			delete []pSkipArr[i];
		delete []pSkipArr;
		srcImg=outImg;
		return PK_SUCCESS;
	}

	int MedF(double **s,int w,int h,int n)
	{
		if(NULL==s||*s==NULL)
			return PK_ERROR_PARAM;
		double**temp=new double*[h+2*(n/2)];
		for(int i=0;i<h+2*(n/2);i++)
			temp[i]=new double[w+2*(n/2)];
		for(int i=0;i<h+2*(n/2);i++)
			for(int j=0;j<w+2*(n/2);j++)
				temp[i][j]=0.0;
		for(int i=0;i<h;i++)
			for(int j=0;j<w;j++)
				temp[i+n/2][j+n/2]=s[i][j];
		for(int i=0;i<h;i++)
			for(int j=0;j<w;j++)
			{
				s[i][j]=0.0;
				for(int r=0;r<n;r++)
					for(int c=0;c<n;c++)
						s[i][j]+=temp[i+r][j+c];
				s[i][j]/=n*n;
			}
			if(temp!=NULL)
			{
				for(int i=0;i<h+2*(n/2);i++)
					if(temp[i]!=NULL)
						delete [] temp[i];
				delete [] temp;
			}
			return PK_SUCCESS;
	}

	int dilation(CpkMat&dest,CpkMat&src,CpkMat&mask)
	{
		CpkMat tmp(src.Row,src.Col,1,src.GetType());
		int *pMask= mask.GetData<int>();
		BYTE* pSrc=src.GetData<BYTE>();
		BYTE* pDest=tmp.GetData<BYTE>();
		for (int i=1; i<src.Row-1; i++)
		{
			for (int j=0; j<src.lineSize; j++)
			{
				bool bPass=true;
				pDest[i*dest.lineSize + j]=0;
				for (int m=0; m<mask.Row; m++)
				{
					for (int n=0; n<mask.Col; n++)
					{
						if(pMask[m*mask.lineSize+ n] == 0)
							continue;
						if (255 == pSrc[(i+2-m-1)*src.lineSize + n-1+j])
						{
							bPass=false;
							pDest[i*dest.lineSize + j]=255;
							break;
						} 
					}
				}
			}
		}
		dest=tmp;
		return PK_SUCCESS;
	}

	int erosion(CpkMat&dest,CpkMat&src,CpkMat&mask)
	{
		CpkMat tmp(src.Row,src.Col,1,src.GetType());
		int *pMask= mask.GetData<int>();
		BYTE* pSrc=src.GetData<BYTE>();
		BYTE* pDest=tmp.GetData<BYTE>();
		for (int i=1; i<src.Row-1; i++)
		{
			for (int j=0; j<src.lineSize; j++)
			{
				bool bPass=true;
				pDest[i*dest.lineSize + j]=255;
				for (int m=0; m<mask.Row; m++)
				{
					for (int n=0; n<mask.Col; n++)
					{
						if(pMask[m*mask.lineSize+ n] == 0)
							continue;
						if (0 == pSrc[(i+2-m-1)*src.lineSize + n-1+j])
						{
							bPass=false;
							pDest[i*dest.lineSize + j]=0;
							break;
						} 
					}
				}
			}
		}
		dest=tmp;
		return PK_SUCCESS;
	}

	int EqualizeHist(CpkMat& dest,CpkMat& src)
	{
		if(src.GetType()!=CpkMat::DATA_BYTE)
			return PK_NOT_ALLOW_OPERATOR;
		CpkMat tmp(src.Row,src.Col,src.Depth,CpkMat::DATA_BYTE);
		float levels[256]={0};
		int   map[256]={0};
		BYTE* pSrc=src.GetData<BYTE>();
		for(int i=0;i<src.Row*src.lineSize;i++)
			++levels[pSrc[i]];

		for(int i=0;i<src.Row*src.lineSize;i++)
		{
			if(pSrc[i]<9)
				i=i;
		}

		int min=0,max=0;
		for(int i=0;i<256;i++)
		{
			if(levels[i]!=0)
			{
				min=i;
				break;
			}
		}

		min=1;

		for(int i=255;i>0;i--)
		{
			if(levels[i]!=0)
			{
				max=i;
				break;
			}
		}
		
		for(int i=0;i<256;i++)
		{
			if(i!=0)
				levels[i]=levels[i]/(src.Row*src.lineSize)+levels[i-1];
			else
				levels[i]=levels[i]/(src.Row*src.lineSize);
			map[i]=levels[i]*255+0.5;
		}

		if(min==255)
		{
			dest=src;
			return PK_SUCCESS;
		}

		BYTE* pDest=tmp.GetData<BYTE>();
		for(int i=0;i<src.Row*src.lineSize;i++)
//			pDest[i]=(map[pSrc[i]]-min)*max/(max-min);
//			pDest[i]=(map[pSrc[i]]-min)*255/(255-min);
			pDest[i]=map[pSrc[i]];
		dest=tmp;
		return PK_SUCCESS;
	}

	int createDlg(const char* name)
	{
		mapDlgIt it=g_dlgs.find(name);
		if(it!=g_dlgs.end())
			return PK_ERROR_PARAM;
		int nRet=PK_SUCCESS;
#if _MSC_VER
		IShowImgBaseDlg* pDlg=new CShowImgWinDlg;
		nRet=pDlg->CreateDlg(name);
		g_dlgs[name]=pDlg;
#endif
		return nRet;
	}

	int showDlg(const char* name,CpkMat& img)
	{
		mapDlgIt it=g_dlgs.find(name);
		if(it==g_dlgs.end())
			return PK_ERROR_PARAM;
		return it->second->ShowDlg(img);
	}

	int destroyDlg(const char* name)
	{
		mapDlgIt it=g_dlgs.find(name);
		if(it==g_dlgs.end())
			return PK_ERROR_PARAM;
		int nRet=it->second->DestroyDlg();
		if(nRet!=PK_SUCCESS)
			return nRet;
		delete [] it->second;
		g_dlgs.erase(it);
		return PK_SUCCESS;
	}

	//mean是均值，sigma是方差
	int gaussianNoise(CpkMat& src,CpkMat& dest,double sigma,double mean)
	{
		CpkMat rates=pk::randn(src.Row,src.Col);
		CpkMat tmpImg(src.Row,src.Col,src.Depth,src.GetType());
		for(int row=0;row<src.Row;row++)
		{
			for(int col=0;col<src.Col;col++)
			{
				BYTE* pData=src.GetData<BYTE>()+row*src.lineSize+col*src.Depth;
				BYTE* pTmpData=tmpImg.GetData<BYTE>()+row*tmpImg.lineSize+col*tmpImg.Depth;
				for(int k=0;k<src.Depth;k++)
				{
				  double gaussianValue= rates.at<double>(row,col);
				  while(gaussianValue>1)
					  gaussianValue-=1;
				  while(gaussianValue<-1)
					  gaussianValue+=1;
				  //6是保证高斯噪音在[-6,6]之间
				  int tmp=pData[k]+gaussianValue*sigma*6+mean;
				  if(tmp>255)
					  pTmpData[k]=255;
				  else
				  pTmpData[k]=tmp;
				}
			}
		}
		dest=tmpImg;
		return PK_SUCCESS;
	}

	int saltPepperNoise(CpkMat& src,CpkMat& dest,double rate=0.05)
	{
		if(src.GetType()!=CpkMat::DATA_BYTE)
			return PK_NOT_ALLOW_OPERATOR;
		int size=src.Row*src.lineSize;
		int number=size*rate*2;//宽和高各一条

		CpkMat rates=rand(1,number);

		CpkMat tmp(src.Row,src.Col,src.Depth,src.GetType(),src.GetData<BYTE>());
		BYTE* pData=tmp.GetData<BYTE>();
		double* pRates=rates.GetData<double>();
		for(int i=0;i<number;i+=2)
		{
			int row = pRates[i]*(src.Row-1);  
			int col = pRates[i+1]*(src.Col-1);  
			int index=row*src.lineSize+col*src.Depth;
			for(int j=0;j<src.Depth;j++)
				pData[index+j]=255;
		}
		dest=tmp;
		return PK_SUCCESS;
	}

	int imNoise(CpkMat& src,CpkMat& dest,IMAGE_NOISE type,double param1,double param2)
	{
		int nRet=PK_SUCCESS;
		switch(type)
		{
		case GAUSSIAN:
			nRet=gaussianNoise(src,dest,param1,param2);
			break;
		case SALT_PEPPER:
			if(param1==0)
				nRet=saltPepperNoise(src,dest);
			else
				nRet=saltPepperNoise(src,dest,param1);
			break;
		default:
			nRet=PK_NOT_ALLOW_OPERATOR;

		}
		return nRet;
	}

};

