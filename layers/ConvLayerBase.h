#ifndef CONV_LAYER_BASE_H
#define CONV_LAYER_BASE_H
#include "LayerBase.h"

struct CConvLayerBase
{
	int m_convDim;
	int m_kernelDim;
protected:
	precision* m_convData;
	precision* m_convNoActiveData;
};

#endif