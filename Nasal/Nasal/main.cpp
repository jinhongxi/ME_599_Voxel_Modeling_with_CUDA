#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <npp.h>

#include "kernel.h"
#include "c_func.h"

int main(void)
{
	cudaMalloc(&d_in, imgSize.x*imgSize.y*imgSize.z*sizeof(Npp16u));
	cudaMalloc(&d_out, imgSize.x*imgSize.y*imgSize.z*sizeof(Npp16u));

	importNPP(d_in, imgSize);

	cudaFree(d_in);
	cudaFree(d_out);

	getchar();
	return 0;
}