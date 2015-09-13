#ifndef PK_MEM_H
#define PK_MEM_H
#include "pkdefine.h"

#define PK_MALLOC_ALIGN    16

template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
	return (_Tp*)(((size_t)ptr + n-1) & -n);
}

static inline size_t alignSize(size_t sz, int n)
{
	return (sz + n-1) & -n;
}

void* fastMalloc( size_t size )
{
	BYTE* udata = (BYTE*)malloc(size + sizeof(void*) + PK_MALLOC_ALIGN);
	if(!udata)
		return NULL;
	BYTE** adata = alignPtr((BYTE**)udata + 1, PK_MALLOC_ALIGN);
	adata[-1] = udata;
	return adata;
}

void fastFree(void* ptr)
{
	if(ptr)
	{
		BYTE* udata = ((BYTE**)ptr)[-1];
//		CV_DbgAssert(udata < (uchar*)ptr &&
//			((uchar*)ptr - udata) <= (ptrdiff_t)(sizeof(void*)+CV_MALLOC_ALIGN));
		free(udata);
	}
}


#endif