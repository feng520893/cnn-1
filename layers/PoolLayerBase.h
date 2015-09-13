#ifndef POOL_LAYER_BASE_H
#define POOL_LAYER_BASE_H
#include "LayerBase.h"

struct CPoolLayerBase
{
	CPoolLayerBase():m_poolData(NULL),m_maxIndexData(NULL),m_kernelDim(0),m_preConvDim(0),
		m_poolType(MAX_POOL){};
	int m_kernelDim;
	int m_preConvDim;
	POOL_TYPE m_poolType;
protected:
	precision* m_poolData;
	int*       m_maxIndexData;
};

#endif