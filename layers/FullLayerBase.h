#ifndef FULL_LAYER_BASE_H
#define FULL_LAYER_BASE_H
#include "LayerBase.h"

struct CFullLayerBase
{
	float m_dropRate;
protected:
	precision* m_fullData;
	precision* m_fullNoActiveData;
	precision* m_afterDropWeight;

	float*    m_dropProbability;
};

#endif