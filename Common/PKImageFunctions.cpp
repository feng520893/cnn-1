#include "PKImageFunctions.h"
#include "./Jpeg/Jpeg.h"
#include<cmath>
#include <string>
namespace pk
{
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

	bool RGBtoGrayscale(CpkMat&dest,CpkMat&src)
	{
		CpkMat tmp;
		BYTE* pSrc=src.GetData<BYTE>();
		
		if(src.Depth==1)
		{
			tmp.Resize(src.Row,src.lineSize,1,CpkMat::DATA_BYTE);
			tmp.Col=src.Col;
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
			tmp.Resize(src.Row,nLineByteOut,1,CpkMat::DATA_BYTE);
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
		dest.Resize(tmp.Row,tmp.lineSize,tmp.Depth,tmp.GetType(),tmp.GetData<BYTE>());
		dest.Col=tmp.Col;
		return true;
	}

	int GetRGBchannel(CpkMat&src,CpkMat& colorR,CpkMat&colorG,CpkMat& colorB)
	{
		if(src.Depth<3)
			return PK_NOT_ALLOW_OPERATOR;
		int nLineByteOut=(src.Col+3)/4*4;
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

		return PK_SUCCESS;
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
					if(0<=coordinateX&&coordinateX<src.Col&&coordinateY>=0&&coordinateY<src.Row)
					{
						for(k=0;k<pix;k++)
							pDest[i*lineByteOut+j*pix+k]=pSrc[coordinateY*lineByteIn+coordinateX*pix+k];
					}
					else
					{
						for(k=0;k<pix;k++)
							pDest[i*lineByteOut+j*pix+k]=255;
					}
				}
			}
		}
		else
		{
			for (int i = 0; i < heighOut; i++)
			{
				int tH = (int)(ratioY * i);
				int tH1 = PK_MIN(tH + 1,src.Row - 1);
				float u = (float)(ratioY * i - tH);
				for (int j = 0; j < widthOut; j++)
				{
					int tW = (int)(ratioX * j); 
					int tW1 = PK_MIN(tW + 1,src.Col - 1);
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
		dest=tmp;
		return PK_SUCCESS;
	}

	int UpDown(CpkMat&dest,CpkMat&src)
	{
		CpkMat tmp;
		tmp.Resize(src.Row,src.Col,src.Depth,src.GetType());
		for(int i=0;i<src.Row;i++)
		{
			CpkMat tmp2;
			src.getRow(tmp2,src.Row-i-1);
			tmp.setRowData(i,tmp2);
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
			src.getColData(tmp2,src.Col-i-1);
			tmp.setColData(i,tmp2);
		}
		dest=tmp;
		return PK_SUCCESS;
	}

};

