#include"layerFactory.h"

#include"common.h"

#include"layers/DataLayerBase.h"

#include"layers/GPU/ConvLayer.cuh"
#include"layers/GPU/PoolLayer.cuh"
#include"layers/GPU/InnerProductLayer.cuh"
#include"layers/GPU/DropoutLayer.cuh"
#include"layers/GPU/SoftMaxLayer.cuh"
#include"layers/GPU/sigmoidLayer.cuh"
#include"layers/GPU/tanhLayer.cuh"
#include"layers/GPU/reluLayer.cuh"
#include"layers/GPU/lrnLayer.cuh"
#include"layers/GPU/Accuracy.cuh"

#include"layers/CPU/ConvLayer.h"
#include"layers/CPU/PoolLayer.h"
#include"layers/CPU/InnerProductLayer.h"
#include"layers/CPU/DropoutLayer.h"
#include"layers/CPU/SoftMaxLayer.h"
#include"layers/CPU/sigmoidLayer.h"
#include"layers/CPU/tanhLayer.h"
#include"layers/CPU/reluLayer.h"
#include"layers/CPU/lrnLayer.h"
#include"layers/CPU/Accuracy.h"

namespace openNet
{
	CLayerFactory* CLayerFactory::m_pThis=NULL;
	CLayerFactory* CLayerFactory::initialize()
	{
		if(m_pThis == NULL)
			m_pThis=new CLayerFactory();
		return m_pThis;
	}

	void CLayerFactory::destroy()
	{
		if(m_pThis != NULL)
			delete m_pThis;
	}

	CLayer* CLayerFactory::createLayer(LayerParam& param)
	{
		CLayer* pLayer=NULL;
		if(param.typeName == "DATA")
			pLayer = createDataLayer();
		else if(param.typeName == "CONV")
			pLayer = createConvLayer();
		else if(param.typeName == "POOL")
			pLayer = createPoolLayer(param);
		else if(param.typeName == "INNERPRODUCT")
			pLayer = createInnerProductLayer();
		else if(param.typeName == "DROPOUT")
			pLayer = createDropoutLayer();
		else if(param.typeName == "SOFTMAX")
			pLayer = createSoftMaxLayer();
		else if(param.typeName == "ACCURACY")
			pLayer = createAccuracyLayer();
		else if(param.typeName == "SIGMOID")
			pLayer = createSigmoidLayer();
		else if(param.typeName == "TANH")
			pLayer = createTanhLayer();
		else if(param.typeName == "RELU")
			pLayer = createReluLayer();
		else if(param.typeName == "LRN")
			pLayer = createLRNLayer();
		else
		{
		}
		if( pLayer != NULL)
			pLayer->m_param=param;
		return pLayer;
	}

	CLayer* CLayerFactory::createDataLayer()
	{
		CLayer* pLayer=NULL;
		pLayer=new CDataLayerBase();
		return pLayer;
	}
	
	CLayer* CLayerFactory::createConvLayer()
	{
		CLayer* pLayer=NULL;
#ifndef CPU_ONLY
		if(m_mode==NET_CPU_COMPUTE)
			pLayer=new CConvLayerCPU();
		else
			pLayer=new CConvLayerGPU();
#else
		pLayer=new CConvLayerCPU();
#endif
		return pLayer;
	}
		CLayer* CLayerFactory::createPoolLayer(LayerParam& param)
		{
			CLayer* pLayer=NULL;
#ifndef CPU_ONLY
		if(m_mode==NET_CPU_COMPUTE)
			pLayer=new CPoolLayerCPU(param);
		else
			pLayer=new CPoolLayerGPU(param);
#else
		pLayer=new CPoolLayerCPU();
#endif
		return pLayer;
		}
		CLayer* CLayerFactory::createInnerProductLayer()
		{
			CLayer* pLayer=NULL;
#ifndef CPU_ONLY
		if(m_mode==NET_CPU_COMPUTE)
			pLayer=new CInnerProductLayerCPU();
		else
			pLayer=new CInnerProductLayerGPU();
#else
		pLayer=new CInnerProductLayerCPU();
#endif
		return pLayer;
		}

		CLayer* CLayerFactory::createDropoutLayer()
		{
			CLayer* pLayer=NULL;
#ifndef CPU_ONLY
		if(m_mode==NET_CPU_COMPUTE)
			pLayer=new CDropoutLayerCPU();
		else
			pLayer=new CDropoutLayerGPU();
#else
		pLayer=new CDropoutLayerCPU();
#endif
		return pLayer;
		}

		CLayer* CLayerFactory::createSoftMaxLayer()
		{
			CLayer* pLayer=NULL;
#ifndef CPU_ONLY
		if(m_mode==NET_CPU_COMPUTE)
			pLayer=new CSoftmaxLayerCPU();
		else
			pLayer=new CSoftMaxLayerGPU();
#else
		pLayer=new CFullLayerCPU();
#endif
		return pLayer;
		}

		CLayer* CLayerFactory::createSigmoidLayer()
		{
			CLayer* pLayer=NULL;
#ifndef CPU_ONLY
		if(m_mode==NET_CPU_COMPUTE)
			pLayer=new CSigmoidLayerCPU();
		else
			pLayer=new CSigmoidLayerGPU();
#else
		pLayer=new CSigmoidLayerCPU();
#endif
		return pLayer;
		}

		CLayer* CLayerFactory::createTanhLayer()
		{
			CLayer* pLayer=NULL;
#ifndef CPU_ONLY
		if(m_mode==NET_CPU_COMPUTE)
			pLayer=new CTanhLayerCPU();
		else
			pLayer=new CTanhLayerGPU();
#else
		pLayer=new CTanhLayerCPU();
#endif
		return pLayer;
		}

		CLayer* CLayerFactory::createReluLayer()
		{
			CLayer* pLayer=NULL;
#ifndef CPU_ONLY
		if(m_mode==NET_CPU_COMPUTE)
			pLayer=new CReluLayerCPU();
		else
			pLayer=new CReluLayerGPU();
#else
		pLayer=new CReluLayerCPU();
#endif
		return pLayer;
		}

		CLayer* CLayerFactory::createLRNLayer()
		{
			CLayer* pLayer=NULL;
#ifndef CPU_ONLY
		if(m_mode==NET_CPU_COMPUTE)
			pLayer=new CLRNLayerCPU();
		else
			pLayer=new CLRNLayerGPU();
#else
		pLayer=new CLRNLayerCPU();
#endif
		return pLayer;
		}

		CLayer* CLayerFactory::createAccuracyLayer()
		{
			CLayer* pLayer=NULL;
#ifndef CPU_ONLY
		if(m_mode==NET_CPU_COMPUTE)
			pLayer=new CAccuracyCPU();
		else
			pLayer=new CAccuracyGPU();
#else
		pLayer=new CAccuracyCPU();
#endif
		return pLayer;
		}


};