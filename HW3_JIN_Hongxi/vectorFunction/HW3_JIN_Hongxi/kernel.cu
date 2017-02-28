#include "hw3.h"
#include <stdlib.h>
#include <stdio.h>
#define TPB 1024

void valueAssignment(float *x, int l, float num)
{
	for (int i = 0; i < l; ++i)
	{
		x[i] = num;
	}
}

int kernelSize(int len, int Len)
{
	return ((len + Len - 1) / Len);
}

/*---------------------------------------------------------------------------------------------------------*\
|  INPUTS:                                                                                                  |
|  - x1, x2 = floating-point arrays with the same length;                                                   |
|  - l = length of x1 and x2;                                                                               |
|  - a, b, c, d = function paramters;                                                                       |
|                                                                                                           |
|  OUTPUTS:                                                                                                 |
|  - y = A * x1 .* x2 + B * x1 + C * x2 + D                                                                 |
|                                                                                                           |
|  EXAMPLES:                                                                                                |
|  (a) Scalar multiplication:                                                                               |
|      vec_fun(w2, u2, v2, N, 0, C, 0, 0);  \\ w = C * u                                                    |
|  (b) Component-wise addition:                                                                             |
|      vec_fun(w2, u2, v2, N, 0, 1, 1, 0);  \\ w = u .+ v                                                   |
|  (c) Linear function:                                                                                     |
|      vec_fun(w2, u2, v2, N, 0, C, 0, D);  \\ w = C * u + D                                                |
|  (d) Component-wise multiplication:                                                                       |
|      vec_fun(w2, u2, v2, N, 1, 0, 0, 0);  \\ w = u .* v                                                   |
\*---------------------------------------------------------------------------------------------------------*/
__global__
void vectorParallel(float *d_p, float *d_x1, float *d_x2, int len, float A, float B, float C, float D, int atom)
{
	/*
	y = A * x1 .* x2 + B * x1 + C * x2 + D
	*/
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int s_i = threadIdx.x;
	extern __shared__ float s_out[];

	if (i >= len) { s_out[s_i] = 0.0f; }
	else
	{
		s_out[s_i] = A * d_x1[i] * d_x2[i] + B * d_x1[i] + C * d_x2[i] + D;
	}

	__syncthreads();
	
	if (s_i == 0)
	{
		float sum = 0.0f;
		for (int j = 0; j < blockDim.x; ++j) { sum += s_out[j]; }
		switch (atom)
		{
		case 1: {atomicAdd(d_p, sum); break; }
		case 0: {*d_p += sum; break; }
		default: {break; }
		}
	}
}

void vec_fun(float *t, float *pro, float *x1, float *x2, int l, float a, float b, float c, float d, int atom)
{
	switch (atom)
	{
	case 0: case 1:
	{
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		float *d_in1 = 0;
		float *d_in2 = 0;
		float *d_pro;

		cudaMalloc(&d_in1, l*sizeof(float));
		cudaMalloc(&d_in2, l*sizeof(float));
		cudaMalloc(&d_pro, sizeof(float));

		cudaMemcpy(d_in1, x1, l*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_in2, x2, l*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemset(d_pro, 0.0f, sizeof(float));

		const size_t sMemSize = TPB*sizeof(float);

		vectorParallel << <kernelSize(l, TPB), TPB, sMemSize >> >(d_pro, d_in1, d_in2, l, a, b, c, d, atom);

		cudaMemcpy(pro, d_pro, sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(d_in1);
		cudaFree(d_in2);
		cudaFree(d_pro);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float d_t;
		cudaEventElapsedTime(&d_t, start, stop);
		*t = d_t;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		cudaDeviceReset();
		break;
	}
	case -1:
	{
		clock_t start = clock();
		for (int i = 0; i < l; ++i) { *pro += a * x1[i] * x2[i] + b * x1[i] + c * x2[i] + d; }
		clock_t stop = clock();
		*t = ((float)(stop - start) * 1000) / CLOCKS_PER_SEC;
		break;
	}
	default: {break; }
	}
}