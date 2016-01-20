#include"solve.h"
#include"layerFactory.h"
#include<io.h>

CSolve::CSolve():m_currentStep(0),m_currentIter(0)
{
}

CSolve::~CSolve()
{
	CTools::destroy();
}

int CSolve::init(const char* configPath)
{
	CTools* pTool=CTools::initialize();
	if(pTool==NULL)
		return TOOLS_INIT_FAILE;
	openNet::CLayerFactory* pFactory=openNet::CLayerFactory::initialize();
	if(pFactory == NULL)
		return LAYER_FACTORY_INIT_FAILE;

	int nRet=NET_SUCCESS;
	CReadConfig nReadconfig;
	do
	{
		nRet=nReadconfig.loadConfig(configPath);
		if(nRet!=NET_SUCCESS)
			break;

		m_params=nReadconfig.solveParams;

		//保存路径检测
		if(m_params.netMode != 3)
		{
			if(_access(m_params.saveModelPath.c_str(),0)!=0)
				return NET_FILE_NOT_EXIST;
			if(m_params.saveModelPath.at(m_params.saveModelPath.length()-1)!='/'||
				m_params.saveModelPath.at(m_params.saveModelPath.length()-1)!='\\')
				m_params.saveModelPath+='/';
		}

		if(!CTools::isCUDA())
			m_params.runMode=NET_CPU_COMPUTE;
		else
		{
			if(m_params.runMode==NET_AUTO_COMPUTE||m_params.runMode>NET_GPU_CUDA_COMPUTE)
				m_params.runMode=NET_GPU_CUDA_COMPUTE;
		}

		pFactory->setMode(m_params.runMode);

		std::vector<LayerParam> trainLayerParams;
		std::vector<LayerParam> testLayerParams;
		for(int i=0;i<nReadconfig.layerParams.size();i++)
		{
			if(nReadconfig.layerParams[i].phase == TRAIN)
				trainLayerParams.push_back(nReadconfig.layerParams[i]);
			else if(nReadconfig.layerParams[i].phase == TEST)
				testLayerParams.push_back(nReadconfig.layerParams[i]);
			else
			{
				trainLayerParams.push_back(nReadconfig.layerParams[i]);
				testLayerParams.push_back(nReadconfig.layerParams[i]);
			}
		}

		m_trainNet.setPhase(TRAIN);
		m_testNet.setPhase(TEST);
		if(m_params.netMode ==3 )
		{
			nRet=m_trainNet.init(trainLayerParams);
			if(nRet!=NET_SUCCESS)
				break;
		}
		else if(m_params.netMode == 2)
		{
			nRet=m_testNet.init(testLayerParams);
			if(nRet!=NET_SUCCESS)
				break;
		}
		else
		{
			nRet=m_trainNet.init(trainLayerParams);
			if(nRet!=NET_SUCCESS)
				break;
			nRet=m_testNet.init(testLayerParams);
			if(nRet!=NET_SUCCESS)
				break;
			m_trainNet.shareDataTo(m_testNet);
		}
	}while(0);

	openNet::CLayerFactory::destroy();

	return nRet;
}

int CSolve::run()
{
	int nRet=NET_SUCCESS;
	if(m_params.netMode == 3)
	{
		double r=r=m_trainNet.computeNumericalGradient();
		printf("梯度误差:%e\n",r);
	}
	else if(m_params.netMode == 2)
	{
		std::string path=m_params.saveModelPath;
		path+="iterFin.dat";
		nRet=m_testNet.load(path.c_str());
		if(nRet!=NET_SUCCESS)
		{
			printf("找不到权值数据所在文件或文件有问题，请重新生成!");
			return nRet;
		}
		nRet=test();
	}
	else
	{
/*		if(imgPreDeal==1)
		{
			pk::CPCA pca;
			trainImgs=trainImgs.Transpose();

			CpkMat tmpMat;
			pk::avg(tmpMat,trainImgs,pk::DATA_COLS);
			trainImgs=pk::subVec(trainImgs,tmpMat,pk::DATA_ROWS);

			pca.run(trainImgs,pk::DATA_COLS,pk::CPCA::ZCA_WHITE);
			sprintf(dataPath,_T("%sdata\\pca.dat"),exePath);
			pca.saveData(dataPath);

			pca.project(trainImgs,pk::DATA_COLS);
			trainImgs=trainImgs.Transpose();
		}*/

		if(m_params.netMode == 1 )
		{
			std::string path=m_params.saveModelPath;
			path+="iterFin.dat";
			nRet=m_trainNet.load(path.c_str());
			if(nRet!=NET_SUCCESS)
			{
				printf("找不到权值数据所在文件或文件有问题，请重新生成!");
				return nRet;
			}
		}

		nRet=train();
	}
	return nRet;
}

int CSolve::train()
{
	int nRet=NET_SUCCESS;
	double cost=0.0;
	for(m_currentIter = 0;m_currentIter<m_params.maxIter;m_currentIter++)
	{
		float lr=getLearningRate();
		cost=m_trainNet.step();
		nRet=m_trainNet.updateWeights(m_params.momentum,lr);
		if(nRet!=NET_SUCCESS)
			break;
		
		if(m_params.display&&(m_currentIter+1)%m_params.display == 0)
			printf("第%d次迭代误差:%lf 学习率:%lf\n",m_currentIter+1,cost,m_params.baseLR);

		if(m_params.saveInterval!=0&&(m_currentIter+1)%m_params.saveInterval == 0)
		{
			char tmp[256]={0};
			sprintf(tmp,"iter_%d.dat",m_currentIter+1);
			std::string savePath=m_params.saveModelPath;
			savePath+=tmp;
			m_trainNet.save(savePath.c_str());
		}

		if(m_params.testInterval != 0 &&(m_currentIter+1)%m_params.testInterval == 0)
		{
			nRet=test();
			if(nRet!=NET_SUCCESS)
				break;
		}
	}
	std::string path=m_params.saveModelPath;
	path+="iterFin.dat";
	return m_trainNet.save(path.c_str());
}

int CSolve::test()
{
	for(int i=0;i<m_params.testIter;i++)
	{
		m_testNet.step();
		printf("process:%.2lf\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b",100.0*i/m_params.testIter);
	}

	Blob<precision>* pAccucary=m_testNet.getAccuracy();
	CLayer*  pLayer=m_testNet.getLayer("DATA");
	int total=pLayer->m_param.dataParam.batchSize*m_params.testIter;
	if(m_params.runMode==NET_CPU_COMPUTE)
	{
		printf("accuracy:%.2lf\n",100.0*pAccucary->cpuData[0]/total);
		pAccucary->cpuData[0]=0;
	}
	else
	{
		precision accuracy=0.0;
		cudaMemcpy(&accuracy,pAccucary->gpuData,sizeof(precision),cudaMemcpyDeviceToHost);
		printf("accuracy:%.2lf\n",accuracy/total*100);
		cudaMemset(pAccucary->gpuData,0,sizeof(precision));
	}

	return PK_SUCCESS;
}


precision CSolve::getLearningRate()
{
  precision rate;
  const std::string& lr_policy = m_params.LR_Policy;
  if (lr_policy == "FIXED") 
  {
    rate = m_params.baseLR;
  } 
  else if (lr_policy == "STEP") 
  {
    m_currentStep = m_currentIter / m_params.stepSize;
    rate = m_params.baseLR *pow(m_params.gamma,m_currentStep);
  } 
  else if (lr_policy == "EXP") 
  {
    rate = m_params.baseLR * pow(m_params.gamma,m_currentIter);
  } 
  else if (lr_policy == "INV") 
  {
    rate = m_params.baseLR *
        pow(float(1.0) + m_params.gamma * m_currentIter, -m_params.power);
  }
  else if (lr_policy == "MULTISTEP") 
  {
/*    if (this->current_step_ < this->param_.stepvalue_size() &&
          this->iter_ >= this->param_.stepvalue(this->current_step_)) 
	{
      this->current_step_++;
      LOG(INFO) << "MultiStep Status: Iteration " <<
      this->iter_ << ", step = " << this->current_step_;
    }
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);*/
  } 
  else if (lr_policy == "POLY") 
  {
    rate = m_params.baseLR * pow(float(1.) -
		(float(m_currentIter) / float(m_params.maxIter)),
        m_params.power);
  } 
  else if (lr_policy == "SIGMOID") 
  {
    rate = m_params.baseLR * (precision(1.) /
        (precision(1.) + exp(-m_params.gamma * (precision(m_currentIter) -
		precision(this->m_params.stepSize)))));
  } 
  else 
  {
 //   LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  return rate;
}