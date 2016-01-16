#ifndef NET_DEFINE
#define NET_DEFINE

#ifndef FLOAT_TYPE
typedef double precision;
#else
typedef float precision;
#endif

#define NET_AUTO_COMPUTE        0
#define NET_CPU_COMPUTE         1
#define NET_GPU_CUDA_COMPUTE    2

/*******************************
********下面为错误值定义********
*******************************/

#define NET_SUCCESS                0
#define NET_UNKNOW_MESS           -1
#define NET_CREATE_LAYER_FAILE    -2
#define NET_INPUT_SETTING_ERROR   -3
#define NET_OUTPUT_SETTING_ERROR  -4
#define NET_SRC_NOT_EQUAL_DEST    -5

#define NET_NOT_ALLOW_OPERATOR    -10
#define NET_FILE_NOT_EXIST        -11

#define TOOLS_NOT_INIT            -20
#define TOOLS_INIT_FAILE          -21 
#define LAYER_FACTORY_INIT_FAILE  -22


#endif
