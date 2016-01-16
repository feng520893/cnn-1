#include"LRNLayer.h"

int CLRNLayerCPU::setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs)
{
	DL_ASSER(m_param.lrnParam.localSize%2!=0);
	m_prePad = (m_param.lrnParam.localSize-1)/2;
	if(m_param.lrnParam.normRegionType == ACROSS_CHANNELS)
	{
		m_scales.create(inputs[0]->num,inputs[0]->dataChannel,inputs[0]->dimHeight,inputs[0]->dimWidth);
		outputs[0]->create(inputs[0]->num,inputs[0]->dataChannel,inputs[0]->dimHeight,inputs[0]->dimWidth);
	}
	return 0;
}

precision CLRNLayerCPU::feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	int nRet=NET_SUCCESS;
	if(m_param.lrnParam.normRegionType == ACROSS_CHANNELS)
		nRet=crossChannelForward(bottoms,tops);
	return nRet;
}

int CLRNLayerCPU::crossChannelForward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	int localSize=m_param.lrnParam.localSize;
	precision alphaOver=m_param.lrnParam.alpha/localSize;
	for (int i = 0; i < m_scales.size(); ++i) 
		m_scales.cpuData[i] = 1;
	Blob<precision> paddedBottoms(1,bottoms[0]->dataChannel+localSize-1,bottoms[0]->dimHeight,bottoms[0]->dimWidth);
	for(int n=0;n<bottoms[0]->num;++n)
	{
		int dataSize=bottoms[0]->size()/bottoms[0]->num;
		int dimSize= bottoms[0]->dimWidth*bottoms[0]->dimHeight;
		for(int j=0;j<dataSize;j++)
			paddedBottoms.cpuData[m_prePad*dimSize+j]=pow(bottoms[0]->cpuData[n*dataSize+j],2);
		for(int c=0;c<localSize;c++)
		{
			for(int j=0;j<dimSize;j++)
				m_scales.cpuData[n*dataSize+j]+=alphaOver*paddedBottoms.cpuData[c*dimSize+j];
		}

		//加头去尾，计算其他剩下的channel
		for(int c=1;c<bottoms[0]->dataChannel;c++)
		{
			memcpy(m_scales.cpuData+n*dataSize+c*dimSize,m_scales.cpuData+n*dataSize+(c-1)*dimSize,sizeof(precision)*dimSize);
			//加头
			for(int j=0;j<dimSize;j++)
				m_scales.cpuData[n*dataSize+c*dimSize+j]+=alphaOver*paddedBottoms.cpuData[(c+localSize-1)*dimSize+j];
			//去尾
			for(int j=0;j<dimSize;j++)
				m_scales.cpuData[n*dataSize+c*dimSize+j]-=alphaOver*paddedBottoms.cpuData[(c-1)*dimSize+j];
		}
	}

	precision beat=m_param.lrnParam.beat;
	for(int i=0;i<m_scales.size();i++)
		tops[0]->cpuData[i]=bottoms[0]->cpuData[i]*pow(m_scales.cpuData[i],-beat);
	return NET_SUCCESS;
}

int CLRNLayerCPU::backpropagation(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms)
{
	int nRet=NET_SUCCESS;
	if(m_param.lrnParam.normRegionType == ACROSS_CHANNELS)
		nRet=crossChannelBack(tops,propagateDown,bottoms);
	return nRet;
}

int CLRNLayerCPU::crossChannelBack(std::vector<Blob<precision>*>& tops,std::vector<bool>& propagateDown,std::vector<Blob<precision>*>& bottoms)
{
	int channels=bottoms[0]->dataChannel;
	precision beat=m_param.lrnParam.beat;
	int localSize=m_param.lrnParam.localSize;
	 Blob<precision> paddedRatio(1, channels + localSize - 1,bottoms[0]->dimHeight,bottoms[0]->dimWidth);
	 Blob<precision> accumRatio(1, 1,bottoms[0]->dimHeight,bottoms[0]->dimWidth);

	 precision cacheRatioValue = 2 * m_param.lrnParam.alpha * beat / localSize;
	 for(int i=0;i<m_scales.size();i++)
		 bottoms[0]->cpuDiff[i]=tops[0]->cpuDiff[i]*pow(m_scales.cpuData[i],-beat);

	 for (int n = 0; n < bottoms[0]->num; ++n) 
	 {
		 int dataSize=bottoms[0]->size()/bottoms[0]->num;
		 int dimSize= bottoms[0]->dimHeight*bottoms[0]->dimWidth;
		 for(int i=0;i<dataSize;i++)
			 paddedRatio.cpuData[m_prePad*dimSize+i]=tops[0]->cpuDiff[n*dataSize+i]*tops[0]->cpuData[n*dataSize+i]/m_scales.cpuData[n*dataSize+i];
		 memset(accumRatio.cpuData,0,sizeof(precision)*accumRatio.size());

		 for (int c = 0; c < localSize - 1; ++c) 
		 {
			 for(int i=0;i<dimSize;i++)
				 accumRatio.cpuData[i]+=paddedRatio.cpuData[c*dimSize+i];
		 }
		 for (int c = 0; c < bottoms[0]->dataChannel; ++c) 
		 {
			 for(int i=0;i<dimSize;i++)
			 {
				 accumRatio.cpuData[i]+= paddedRatio.cpuData[(c+localSize-1)*dimSize+i];
			 }
			 for(int i=0;i<dimSize;i++)
			 {
				 accumRatio.cpuDiff[i]=bottoms[0]->cpuData[n*dataSize+c*dimSize+i]*accumRatio.cpuData[i];
			 }
			 for(int i=0;i<dimSize;i++)
			 {
				 bottoms[0]->cpuDiff[n*dataSize+c*dimSize+i]-=cacheRatioValue*accumRatio.cpuDiff[i];
			 }
			 for(int i=0;i<dimSize;i++)
			 {
				 accumRatio.cpuData[i]-=paddedRatio.cpuData[c*dimSize+i];
			 }
		 }
	 }
	return NET_SUCCESS;
}