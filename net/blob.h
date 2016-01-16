#ifndef PK_BLOB
#define PK_BLOB

#ifndef CPU_ONLY
#include<cuda_runtime.h>
#endif

#include"DLerror.h"
template<typename T>
struct Blob
{
	Blob()
	{
		num=dataChannel=dimWidth=dimHeight=0;
		m_refCount=cpuData=gpuData=NULL;
		cpuDiff=gpuDiff=NULL;
	}

	Blob(int num,int channel,int height,int width)
	{
		create(num,channel,height,width);
	}

	~Blob()
	{
		destroy();
	}

	int create(int num,int dataChannel,int dimHeight,int dimWidth)
	{
		cpuData=new T[num*dataChannel*dimHeight*dimWidth+sizeof(T)];
		DL_ASSER(cpuData!=NULL);
        memset(cpuData,0,sizeof(T)*num*dataChannel*dimHeight*dimWidth+sizeof(T));
		m_refCount=cpuData+num*dataChannel*dimHeight*dimWidth;

		cpuDiff=new T[num*dataChannel*dimHeight*dimWidth];
		DL_ASSER(cpuData!=NULL);
		memset(cpuDiff,0,sizeof(T)*num*dataChannel*dimHeight*dimWidth);

		cudaError_t cudaStat=cudaMalloc((void**)&gpuData,sizeof(T)*num*dataChannel*dimHeight*dimWidth);
		CUDA_ERROR(cudaStat);
		cudaStat=cudaMemset(gpuData,0,sizeof(T)*num*dataChannel*dimHeight*dimWidth);
		CUDA_ERROR(cudaStat);

		cudaStat=cudaMalloc((void**)&gpuDiff,sizeof(T)*num*dataChannel*dimHeight*dimWidth);
		CUDA_ERROR(cudaStat);
		cudaStat=cudaMemset(gpuDiff,0,sizeof(T)*num*dataChannel*dimHeight*dimWidth);
		CUDA_ERROR(cudaStat);

		this->num=num;
		this->dataChannel=dataChannel;
	    this->dimWidth=dimWidth;
	    this->dimHeight=dimHeight;
		(*m_refCount)++;
		return 0;
	}

	int destroy()
	{
		if(m_refCount!=NULL&&--(*m_refCount)>0)
			return 0;

		if(cpuData!=NULL)
		{
			delete [] cpuData;
			cpuData=NULL;
		}
		if(gpuData!=NULL)
		{
			cudaFree(gpuData);
			gpuData=NULL;
		}

		if(cpuDiff!=NULL)
		{
			delete [] cpuDiff;
			cpuDiff=NULL;
		}
		if(gpuDiff!=NULL)
		{
			cudaFree(gpuDiff);
			gpuDiff=NULL;
		}

		return 0;
	}

	template<typename T>
	int copyFromCpuData(T* cpuData,unsigned int dataLeng)
	{
		memcpy(this->cpuData,cpuData,sizeof(T)*dataLeng);

		cudaError_t cudaStat=cudaMemcpy(gpuData,cpuData,sizeof(T)*dataLeng,cudaMemcpyHostToDevice);
		CUDA_ERROR(cudaStat);
		return 0;
	}

	template<typename T>
	int copyFromGpuData(T* gpuData,unsigned int dataLeng)
	{
		cudaError_t cudaStat=cudaMemcpy(cpuData,gpuData,sizeof(T)*dataLeng,cudaMemcpyDeviceToHost);
		CUDA_ERROR(cudaStat);

		cudaStat=cudaMemcpy(this->gpuData,gpuData,sizeof(T)*dataLeng,cudaMemcpyDeviceToDevice);
		CUDA_ERROR(cudaStat);
		return 0;
	}

	template<typename T>
	int shareDataFrom(Blob<T>* srcData)
	{
		if(this->gpuData!=NULL||this->cpuData!=NULL)
			this->destroy();
		this->num=srcData->num;
		this->dataChannel=srcData->dataChannel;
	    this->dimWidth=srcData->dimWidth;
	    this->dimHeight=srcData->dimHeight;
		this->cpuData=srcData->cpuData;
		this->gpuData=srcData->gpuData;
		this->cpuDiff=srcData->cpuDiff;
		this->gpuDiff=srcData->gpuDiff;
		this->m_refCount=srcData->m_refCount;
		(*m_refCount)++;
		return 0;
	}

	unsigned int size() const{return num*dataChannel*dimHeight*dimWidth;}

	inline unsigned int offset(const int num,const int channel=0,const int height=0,const int width=0) const
	{
		return ((num*this->datatChannel+channel)*this->dimHeight+height)*this->dimWidth+width;
	}

	int num;
	int dataChannel;
	int dimWidth;
	int dimHeight;

	T*  cpuData;
	T*  gpuData;
	T*  cpuDiff;
	T*  gpuDiff;
	private:
		T* m_refCount;
};

#endif