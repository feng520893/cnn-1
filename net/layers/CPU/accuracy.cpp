#include"Accuracy.h"

int CAccuracyCPU::setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs)
{
	outputs[0]->create(1,1,1,1);
	cudaMalloc((void**)&m_weight,sizeof(precision)*inputs[0]->num);
	return 0;
}

precision CAccuracyCPU::feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops)
{
	for(int i=0;i<bottoms[0]->num;i++)
	{
		precision maxValue=0.0;
		int index=-1;
		for(int j=0;j<bottoms[0]->dataChannel;j++)
		{
			precision tmp=bottoms[0]->cpuData[i*bottoms[0]->dataChannel+j];
			if(maxValue<tmp)
			{
				maxValue=tmp;
				index=j;
			}
		}
		if(index == bottoms[1]->cpuData[i])
			++tops[0]->cpuData[0];
	}
	return NET_SUCCESS;
}