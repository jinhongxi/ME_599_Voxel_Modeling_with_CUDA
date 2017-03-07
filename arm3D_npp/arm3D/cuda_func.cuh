#ifndef CUDAFUNC_CUH
#define CUDAFUNC_CUH

#include <math.h>
#include <helper_math.h>
#include <npp.h>

int divUp(int a, int A);

__device__
int flatten(int3 i, int3 volSize);

#endif