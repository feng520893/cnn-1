#ifndef COMMON_H
#define COMMON_H
#include"pkdefine.h"
int randn(double* data,int dataSize,float rate=0.1);

void transpose(double** data,int& row,int& col);

#endif