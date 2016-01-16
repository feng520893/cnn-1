#ifndef LAYER_FACTORY_H
#define LYAER_FACTORY_H

#include"layers/LayerBase.h"
namespace openNet
{
	class CLayerFactory
	{
	public:
		static CLayerFactory* initialize();
		static void destroy();
		CLayer* createLayer(LayerParam& param);
		void setMode(int modeType){m_mode=modeType;};
	private:
		CLayerFactory():m_mode(NET_CPU_COMPUTE){};
		~CLayerFactory(){};
		CLayer* createDataLayer();
		CLayer* createConvLayer();
		CLayer* createPoolLayer(LayerParam& param);
		CLayer* createInnerProductLayer();
		CLayer* createDropoutLayer();
		CLayer* createSoftMaxLayer();
		CLayer* createSigmoidLayer();
		CLayer* createTanhLayer();
		CLayer* createReluLayer();
		CLayer* createLRNLayer();
		CLayer* createAccuracyLayer();
		static CLayerFactory* m_pThis;
		short m_mode;
	};
};
#endif