#pragma once
	
bool JpegCompression(unsigned char **outbuf,unsigned long&outSize,int width,int height,int bit,int quality,unsigned char *inbuf,unsigned int inSize);

	
bool JpegDecompress(unsigned char **outbuf,unsigned long&outSize,int& outWidth,int& outHeight,int& outBit,unsigned char *inbuf,unsigned int inSize);

