#ifndef ACCURACY_CPU_H
#define ACCURACY_CPU_H
#include"LayerBaseCPU.h"

class CAccuracyCPU : public CLayerBaseCPU
{
public:
	int setup(std::vector<Blob<precision>*>& inputs,std::vector<Blob<precision>*>& outputs);
	precision feedforward(std::vector<Blob<precision>*>& bottoms,std::vector<Blob<precision>*>& tops);
};

#endif