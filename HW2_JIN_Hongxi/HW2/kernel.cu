#include "hw2.h"
#include <stdlib.h>
#include <stdio.h>
#define TPB 32

/*---------------------------------------------------------------------------------------------------------*\
|  PART 1: "vec_fun" notes                                                                                  |
|                                                                                                           |
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
void vectorKernel(float *d_y, float *d_x1, float *d_x2, float A, float B, float C, float D)
{
	/*
	y = A * x1 .* x2 + B * x1 + C * x2 + D
	*/
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	d_y[i] = A * d_x1[i] * d_x2[i] + B * d_x1[i] + C * d_x2[i] + D;
	printf("%.2f; ", d_y[i]);
}

void vec_fun(float *y, float *x1, float *x2, int l, float a, float b, float c, float d)
{
	float *d_in1 = 0;
	float *d_in2 = 0;
	float *d_out = 0;

	cudaMalloc(&d_in1, l*sizeof(float));
	cudaMalloc(&d_in2, l*sizeof(float));
	cudaMalloc(&d_out, l*sizeof(float));

	cudaMemcpy(d_in1, x1, l*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_in2, x2, l*sizeof(float), cudaMemcpyHostToDevice);

	vectorKernel << <l / TPB, TPB >> >(d_out, d_in1, d_in2, a, b, c, d);

	cudaMemcpy(y, d_out, l*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_in1);
	cudaFree(d_in2);
	cudaFree(d_out);
}

/*---------------------------------------------------------------------------------------------------------*\
|  PART 4: "scaleComputing" notes                                                                           |
|                                                                                                           |
|  INPUTS:                                                                                                  |
|  - x1, x2 = floating-point arrays with the same length;                                                   |
|  - l = length of x1 and x2;                                                                               |
|  - director = defines the output of this function;                                                        |
|                                                                                                           |
|  OUTPUTS:                                                                                                 |
|  - return = (director = 1) inner product of x1 and x2                                                     |
|             (director = 2) reversed inner product of x1 and x2                                            |
|             (director = 3) Euclidean norm of x1 + x2                                                      |
|                                                                                                           |
|  EXAMPLES:                                                                                                |
|  (a) Inner product:                                                                                       |
|      inner_product = scaleComputing(w2, u2, v2, N, 1);  \\ inner product of u2 & v2                       |
|  (b) Revised Inner product:                                                                               |
|      revised_product = scaleComputing(w2, u2, v2, N, 2);  \\ revised inner product of u2 & v2             |
|  (c) Euclidean norm:                                                                                      |
|      euc_norm = scaleComputing(w2, u2, v2, N, 3);  \\ Euclidean norm of w2 = u2 + v2                      |
\*---------------------------------------------------------------------------------------------------------*/
float scaleComputing(float *y, float *x1, float *x2, int l, int director)
{
	float product = 0.0f;

	switch (director)
	{
	case 1:
	{
		vec_fun(y, x1, x2, l, 1, 0, 0, 0);
		for (int j = 0; j < l; ++j) { product = product + y[j]; }
		break;
	}
	case 2:
	{
		for (int j = 0; j < l; ++j) { product = product + x1[j] * x2[l - j - 1]; }
		break;
	}
	case 3:
	{
		vec_fun(y, x1, x2, l, 0, 1, 1, 0);
		for (int j = 0; j < l; ++j) { product = product + y[j] * y[j]; }
		product = sqrt(product);
		break;
	}
	default: break;
	}

	return product;
}

/*---------------------------------------------------------------------------------------------------------*\
|  "valueAssignment" notes                                                                                  |
|                                                                                                           |
|  INPUTS:                                                                                                  |
|  - x = floating-point array to assign uniform values;                                                     |
|  - l = length of x;                                                                                       |
|  - num = the value of the array to assign;                                                                |
\*---------------------------------------------------------------------------------------------------------*/
__global__
void valueKernel(float *d_in, float value)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	d_in[i] = value;
	printf("%.2f; ", d_in[i]);
}

void valueAssignment(float *x, int l, float num)
{
	float *d_x = 0;

	cudaMalloc(&d_x, l*sizeof(float));

	cudaMemcpy(&d_x, x, l*sizeof(float), cudaMemcpyHostToDevice);

	valueKernel << <l / TPB, TPB >> >(d_x, num);

	cudaMemcpy(x, d_x, l*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_x);
}