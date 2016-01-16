#include"tools.h"
#include"DLerror.h"
#include"netDefine.h"
#include<io.h>
#include<algorithm>

CTools* CTools::m_pThis=NULL;

CTools* CTools::initialize()
{
	if(m_pThis != NULL)
		return m_pThis;
	m_pThis=new CTools();
#ifndef CPU_ONLY
	curandStatus_t status=curandCreateGenerator(&m_pThis->m_hGen, CURAND_RNG_PSEUDO_DEFAULT);
	CURAND_ERROR(status);

	status=curandSetPseudoRandomGeneratorSeed(m_pThis->m_hGen,time(NULL));
	CURAND_ERROR(status);

	cublasStatus_t stat = cublasCreate(&m_pThis->m_hCublas);
	CUBLAS_ERROR(stat);

#endif
	return m_pThis;
}
void CTools::destroy()
{
	if(m_pThis)
	{
#ifndef CPU_ONLY
		curandDestroyGenerator(m_pThis->m_hGen);
		cublasDestroy(m_pThis->m_hCublas);
#endif
		delete m_pThis;
		m_pThis=NULL;
	}
}

int CTools::findDirectsOrFiles(std::string direct,std::vector<std::string>& files,const char* extension,bool bOnlyFindDirect)
{
	std::string path=direct+std::string("\\*.*");
	long handle;  

	struct _finddata_t fileinfo;
	handle=_findfirst(path.c_str(),&fileinfo);  
	if(-1==handle)
		return NET_FILE_NOT_EXIST;   
	while(!_findnext(handle,&fileinfo))  
	{  
		if(strcmp(fileinfo.name,".")==0||strcmp(fileinfo.name,"..")==0)
			continue;                                                         
		if(bOnlyFindDirect&&fileinfo.attrib!=_A_SUBDIR)  
			continue;   
		if(extension!=NULL)
		{
			std::string extensionStr=extension;
			std::string fileNameStr=fileinfo.name;
			int index=fileNameStr.rfind('.')+1;
			fileNameStr=fileNameStr.substr(index,fileNameStr.length()-index);
			std::transform(extensionStr.begin(),extensionStr.end(),extensionStr.begin(),toupper);
			std::transform(fileNameStr.begin(),fileNameStr.end(),fileNameStr.begin(),toupper);
			if(extensionStr.find(fileNameStr)==std::string::npos)
				continue;
		}

		std::string destFile=direct+std::string("\\")+std::string(fileinfo.name);  
		files.push_back(destFile);
	}  
	_findclose(handle);  
	return NET_SUCCESS;
}

#ifndef CPU_ONLY

bool CTools::isCUDA()
{
    int count;
    cudaGetDeviceCount(&count);
    if(count == 0)
    {
		//没N卡设备
		 return false;
	}
	int i;
    for(i = 0; i < count; i++) 
	{
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) 
		{
            if(prop.major >= 1) 
                break;
        }
    }
    if(i == count) 
	{
		//当前驱动程序不支持CUDA
        return false;
    }
    cudaSetDevice(i);
    return true;
}

curandStatus_t CTools::cudaRandF(float* data,unsigned int dataSize,RAND_TYPE type,float mean,float stddev)
{
	curandStatus_t state;
	if(type == NORMAL)
	{
		state=curandGenerateUniform(m_pThis->m_hGen,data,dataSize);
	}
	else
		state=curandGenerateNormal(m_pThis->m_hGen,data,dataSize,mean,stddev);
	return state;
}

curandStatus_t CTools::cudaRandD(double* data,unsigned int dataSize,RAND_TYPE type,float mean,float stddev)
{
	curandStatus_t state;
	if(type == NORMAL)
	{
		state=curandGenerateUniformDouble(m_pThis->m_hGen,data,dataSize);
	}
	else
		state=curandGenerateNormalDouble(m_pThis->m_hGen,data,dataSize,mean,stddev);
	return state;
}

void CTools::cudaMatrixMulD(double* x,int rowsX,int colsX,double* y,int rowsY,int colsY,double* z,int colsZ,MAT_MUL_TYPE type)
{
	cublasStatus_t stat=CUBLAS_STATUS_SUCCESS;
	switch(type)
	{
	case NORMAL_XY:
		stat=m_pThis->cudaMatrixDMul(x,rowsX,colsX,y,rowsY,colsY,z,colsZ);
		break;
	case TRANSPOSE_Y:
		stat=m_pThis->cudaMatrixDMulTA(x,rowsX,colsX,y,rowsY,colsY,z,colsZ);
		break;
	case TRANSPOSE_X:
		stat=m_pThis->cudaMatrixDMulTB(x,colsX,y,rowsY,colsY,z,colsZ);
		break;
	default:
		DL_ASSER(false);
	}
	CUBLAS_ERROR(stat);
}

cublasStatus_t CTools::cudaMatrixDMulTA(double* x,int rowsX,int colsX,double* y,int rowsY,int colsY,double* z,int colsZ) 
{  
	cublasStatus_t stat=CUBLAS_STATUS_SUCCESS;
 	double alpha = 1.0; 
 	double beta = 0.0; 
 	stat = cublasDgemm(m_hCublas,  
		               CUBLAS_OP_T, 
					   CUBLAS_OP_N, 
					   rowsY, 
					   rowsX, 
					   colsY, 
					   &alpha, 
					   y, 
					   colsY, 
					   x, 
					   colsX, 
					   &beta, 
					   z, 
					   colsZ); 
 	cudaDeviceSynchronize(); 
 	return stat;
} 

cublasStatus_t CTools::cudaMatrixDMulTB(double * x,int colsX,double* y,int rowsY,int colsY,double* z,int colsZ) 
{  
	cublasStatus_t stat=CUBLAS_STATUS_SUCCESS;
 	double alpha = 1.0; 
 	double beta = 0.0; 
 	stat = cublasDgemm(m_hCublas,  
		               CUBLAS_OP_N, 
					   CUBLAS_OP_T, 
					   colsY, 
					   colsX, 
					   rowsY, 
					   &alpha, 
					   y, 
					   colsY, 
					   x, 
					   colsX, 
					   &beta, 
					   z, 
					   colsZ); 
 	cudaDeviceSynchronize(); 
 	return stat;
} 

cublasStatus_t CTools::cudaMatrixDMul(double* x,int rowsX,int colsX,double* y,int rowsY,int colsY,double* z,int colsZ) 
{  
	cublasStatus_t stat=CUBLAS_STATUS_SUCCESS;

 	double alpha = 1.0; 
 	double beta = 0.0; 
 	stat = cublasDgemm(m_hCublas,  
 		               CUBLAS_OP_N, 
					   CUBLAS_OP_N, 
					   colsY, 
					   rowsX, 
					   rowsY, 
					   &alpha, 
					   y, 
					   colsY, 
					   x, 
					   colsX, 
					   &beta, 
					   z, 
					   colsZ); 
 	cudaDeviceSynchronize(); 
	return stat;
} 
#endif