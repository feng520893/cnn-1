#ifndef PKDEFINE_H
#define PKDEFINE_H
#include <cmath>
#define NULL 0
typedef unsigned long DWORD;
typedef unsigned short WORD;
typedef unsigned char BYTE;
typedef long LONG;

#ifndef PK_MAX
#define PK_MAX(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef PK_MIN
#define PK_MIN(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#define INIT_LENG 512*1024

#define PK_SUCCESS                0
#define PK_NOT_FILE              -1
#define PK_ALLOCATE_MEMORY_FAIL  -2
#define PK_READ_FILE_FAIL        -3
#define PK_ERROE_FORMAT          -4
#define PK_NOT_ALLOW_OPERATOR    -5
#define PK_FAIL                  -6
#define PK_OPEN_FILE_ERROR       -7
#define PK_CHECK_NOT_FACE        -8
#define PK_NOT_REAL_FACE         -9
#define PK_NOT_DATA              -10
#define PK_NOT_SUPPORT_FORMAT    -11
#define PK_NOT_INIT              -12
#define PK_ERROR_PARAM           -13


//网络部分
#define PK_NET_NOT_DATA        -100
#define PK_NET_ERROR_MARK      -101
#define PK_NET_ERROR_MD5       -102

//卷积神经网络
#define PK_CNN_NOT_INIT        -200


struct MinSGD
{
	unsigned short epoches;
	unsigned short minibatch;
	float alpha;
	float momentum;
};
namespace pk
{
	struct vecDistance
	{
		int id;
		int start;
		int end;
	};

	struct Rect
	{
		int x;
		int y;
		int width;
		int height;
	};
};

#endif