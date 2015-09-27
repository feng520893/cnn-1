#include "Jpeg.h"
#include <cstring>

extern "C" 
{
#include "jpeglib.h"
}
#pragma comment(lib, "jpeg.lib") 

bool g_err=0;


METHODDEF(void) JpegErrorExit(j_common_ptr cinfo)
{
	if(g_err==0)
	{
		(*cinfo->err->output_message) (cinfo);
		g_err=1;
	}
}

bool JpegDecompress(unsigned char **outbuf,unsigned long&outSize,int& outWidth,int& outHeight,int& outBit,unsigned char *inbuf,unsigned int inSize)
{
	if(*outbuf!=NULL)
	{
		outSize=0;
		return false;
	}
	g_err=0;
	struct jpeg_error_mgr pub;
	struct jpeg_decompress_struct cinfo;
	cinfo.err = jpeg_std_error(&pub);
	pub.error_exit =JpegErrorExit;

	jpeg_create_decompress(&cinfo);
	jpeg_mem_src(&cinfo,inbuf,inSize);
	jpeg_read_header(&cinfo, TRUE);

	if(g_err)
		return false;
	jpeg_start_decompress(&cinfo);
	if(g_err)
		return false;

	outBit=cinfo.output_components;
	outWidth=cinfo.output_width;
	outHeight=cinfo.output_height;

	int row_stride =(outWidth * outBit+3)/4*4;
	outSize=row_stride*outHeight;


	JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);
	if(g_err)
		return false;

	*outbuf=new unsigned char[outSize];
	unsigned char* pDest=*outbuf+row_stride*(outHeight-1);

	while (cinfo.output_scanline < cinfo.output_height)
	{

		jpeg_read_scanlines(&cinfo,buffer,1);
		memcpy(pDest,*buffer,row_stride);
		pDest -= row_stride;
	}

	if(outBit!=1)
	{
		pDest=*outbuf;
		for (unsigned long i=0;i<outHeight;i++)
		{
			for (unsigned long j=0;j<outWidth*outBit;j+=outBit)
			{
				unsigned char r=pDest[j+0];
				unsigned char b=pDest[j+2];
				pDest[j+0]=b;
				pDest[j+2]=r;
			}
			pDest+=row_stride;
		}
	}


	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	return true;
}

bool JpegCompression(unsigned char **outbuf,unsigned long&outSize,int width,int height,int bit,int quality,unsigned char *inbuf,unsigned int inSize)
{
	if(*outbuf!=NULL)
	{
		outSize=0;
		return false;
	}
	if(bit<8)
		return false;
	bit/=8;
	g_err=0;
	struct jpeg_error_mgr pub;
	struct jpeg_compress_struct cinfo;
	cinfo.err = jpeg_std_error(&pub);
	pub.error_exit =JpegErrorExit;

	jpeg_create_compress(&cinfo);

	jpeg_mem_dest(&cinfo,outbuf,&outSize);

	cinfo.image_width = width;    // Ϊͼ�Ŀ�͸ߣ���λΪ���� 
	cinfo.image_height = height;
	cinfo.input_components = bit;   // �ڴ�Ϊ1,��ʾ�Ҷ�ͼ�� ����ǲ�ɫλͼ����Ϊ3 

	unsigned char* data = NULL;

	int row_stride=(width*bit+3)/4*4;

	if(bit==1)
		cinfo.in_color_space = JCS_GRAYSCALE;
	else
	{
		cinfo.in_color_space=JCS_RGB;
		//RGB˳�����  

		data=new unsigned char[row_stride*height];
		for (int x = 0 ; x< width;x++)
		{
			for (int y = 0 ; y < height; y++)
			{			
				//ȡ(x,y)�ĵ�����ֵ�ڴ�����
				unsigned char* ptr = inbuf + (height-y-1)*row_stride + x * bit;				
				int b = *ptr++;
				int g = *ptr++;
				int r = *ptr++;	
				//��ֵRGB������data��(RGB��bmp�е�RGB˳���෴)
				unsigned char* p2 = data + (height-y-1)*row_stride + x * bit;	
				*p2++ = r;		
				*p2++ = g;		
				*p2++ = b;		
			}
		}
	}

	jpeg_set_defaults(&cinfo); 
	jpeg_set_quality (&cinfo,quality, true);
	jpeg_start_compress(&cinfo, TRUE);

	JSAMPROW row_pointer[1];   // һ��λͼ

	unsigned char* pSrc=NULL;
	if(bit!=1)
		pSrc=data+row_stride*(cinfo.image_height-1);
	else
		pSrc=inbuf+row_stride*(cinfo.image_height-1);

	while (cinfo.next_scanline <cinfo.image_height) 
	{
		//		row_pointer[0] = & inbuf[cinfo.next_scanline * row_stride];
		row_pointer[0]=pSrc-cinfo.next_scanline * row_stride;//libJpeg���ɵ�ͼƬ�Ƿ�������,���������ٷ���ȥ��ȡ
		(void)jpeg_write_scanlines(&cinfo, row_pointer, 1);
	}
	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);
	if(data)
		delete [] data;
	return true;
}