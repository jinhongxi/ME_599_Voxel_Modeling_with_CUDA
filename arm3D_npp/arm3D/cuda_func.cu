#include "cuda_func.cuh"

int divUp(int a, int A)
{
	return (a + A - 1) / A;
}

__device__
int flatten(int3 i, int3 volSize)
{
	return (i.z*volSize.x*volSize.y + i.y*volSize.z + i.x);
}