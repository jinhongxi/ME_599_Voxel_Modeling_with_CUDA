#include "CImg.h"
#include "kernel.h"
#include "cuda_func.cuh"
#include <stdio.h>

#define TX2 32
#define TY2 32
#define TX 8
#define TY 8
#define TZ 8
#define STD 0.01f
#define DNE 3

void importLauncher(uchar4 *d_in, int3 volSize)
{
	uchar4 *img = (uchar4*)malloc(volSize.x*volSize.y*volSize.z*sizeof(uchar4));
	for (int s = 0; s < volSize.z; ++s)
	{
		char importFile[28];
		sprintf(importFile, "Color_arm_1/arm_%i.bmp", s + 1);
		cimg_library::CImg<unsigned char>image(importFile);
		for (int r = 0; r < volSize.y; ++r)
		{
			for (int c = 0; c < volSize.x; ++c)
			{
				int i = h_flatten(c, r, s, volSize.x, volSize.y, volSize.z);
				img[i].x = image(c, r, 0);
				img[i].y = image(c, r, 1);
				img[i].z = image(c, r, 2);
				img[i] = colorConvert(img[i]);
			}
		}
	}
	cudaMemcpy(d_in, img, volSize.x*volSize.y*volSize.z*sizeof(uchar4), cudaMemcpyHostToDevice);
	free(img);
}

void boneKernelLauncher(uchar4 *d_in, float *d_vol, int3 volSize)
{
	float *buffer = 0, *d_norm = 0;
	int *d_min = 0, *d_max = 0;
	int h_min = 0, h_max = 0;
	float h_norm = STD*STD*(volSize.x*volSize.y*volSize.z - 1) + 1;
	cudaMalloc(&buffer, volSize.x*volSize.y*volSize.z*sizeof(float));
	cudaMalloc(&d_norm, sizeof(float));
	cudaMalloc(&d_min, sizeof(int));
	cudaMalloc(&d_max, sizeof(int));
	cudaMemset(d_min, 0, sizeof(int));
	cudaMemset(d_max, 0, sizeof(int));
	dim3 blockSize(TX, TY, TZ);
	dim3 gridSize(divUp(volSize.x, TX), divUp(volSize.y, TY), divUp(volSize.z, TZ));
	size_t sharedSize = (size_t)((TX + 2)*(TY + 2)*(TZ + 2))*sizeof(float);

	makeBufferKernel << <gridSize, blockSize, sharedSize >> >(d_in, d_vol, 'b', volSize);

	while (h_norm > STD*STD*(volSize.x*volSize.y*volSize.z - 1))
	{
		plusBufferKernel << <gridSize, blockSize, sharedSize >> >(buffer, d_vol, volSize);
		cudaMemset(d_norm, 0, sizeof(float));
		normBufferKernel << <gridSize, blockSize, sharedSize >> >(d_norm, d_vol, buffer, volSize);
		cudaMemcpy(&h_norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
		copyBufferKernel << <gridSize, blockSize >> >(d_vol, buffer, volSize);
	}

	h_norm = STD*STD*(volSize.x*volSize.y*volSize.z - 1) + 1;
	while (h_norm > STD*STD*(volSize.x*volSize.y*volSize.z - 1))
	{
		avgBufferKernel << <gridSize, blockSize, sharedSize >> >(buffer, d_vol, volSize);
		cudaMemset(d_norm, 0, sizeof(float));
		normBufferKernel << <gridSize, blockSize, sharedSize >> >(d_norm, d_vol, buffer, volSize);
		cudaMemcpy(&h_norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
		copyBufferKernel << <gridSize, blockSize >> >(d_vol, buffer, volSize);
	}

	h_norm = STD*STD*(volSize.x*volSize.y*volSize.z - 1) + 1;
	while (h_norm > STD*STD*(volSize.x*volSize.y*volSize.z - 1))
	{
		extremeBufferKernel << <gridSize, blockSize, sharedSize >> >(buffer, d_vol, volSize);
		cudaMemset(d_norm, 0, sizeof(float));
		normBufferKernel << <gridSize, blockSize, sharedSize >> >(d_norm, d_vol, buffer, volSize);
		cudaMemcpy(&h_norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
		copyBufferKernel << <gridSize, blockSize >> >(d_vol, buffer, volSize);
	}

	size_t sharedSize_i = (size_t)(TX*TY*TZ)*sizeof(int);
	findExmKernel << <gridSize, blockSize, sharedSize_i >> >(d_max, d_min, d_vol, volSize);
	fillKernel << <gridSize, blockSize, sharedSize >> >(d_max, d_min, d_vol, buffer, volSize);
	copyBufferKernel << <gridSize, blockSize >> >(buffer, d_vol, volSize);
	findBondaryKernel << <gridSize, blockSize, sharedSize >> >(d_vol, buffer, volSize);

	h_norm = STD*STD*(volSize.x*volSize.y*volSize.z - 1) + 1;
	while (h_norm > STD*STD*(volSize.x*volSize.y*volSize.z - 1))
	{
		plusBufferKernel << <gridSize, blockSize, sharedSize >> >(buffer, d_vol, volSize);
		cudaMemset(d_norm, 0, sizeof(float));
		normBufferKernel << <gridSize, blockSize, sharedSize >> >(d_norm, d_vol, buffer, volSize);
		cudaMemcpy(&h_norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
		copyBufferKernel << <gridSize, blockSize >> >(d_vol, buffer, volSize);
	}
	mapBufferKernel << <gridSize, blockSize >> >(d_in, d_vol, 'b', volSize);
	//showBufferKernel << <gridSize, blockSize >> >(d_in, d_vol, volSize);

	cudaFree(buffer);
	cudaFree(d_norm);
	cudaFree(d_min);
	cudaFree(d_max);
}

void muscleKernelLauncher(uchar4 *d_in, float *d_vol, int3 volSize)
{
	float *buffer = 0, *d_norm = 0;
	int *d_min = 0, *d_max = 0;
	int h_min = 0, h_max = 0;
	float h_norm = STD*STD*(volSize.x*volSize.y*volSize.z - 1) + 1;
	cudaMalloc(&buffer, volSize.x*volSize.y*volSize.z*sizeof(float));
	cudaMalloc(&d_norm, sizeof(float));
	cudaMalloc(&d_min, sizeof(int));
	cudaMalloc(&d_max, sizeof(int));
	cudaMemset(d_min, 0, sizeof(int));
	cudaMemset(d_max, 0, sizeof(int));
	dim3 blockSize(TX, TY, TZ);
	dim3 gridSize(divUp(volSize.x, TX), divUp(volSize.y, TY), divUp(volSize.z, TZ));
	size_t sharedSize = (size_t)((TX + 2)*(TY + 2)*(TZ + 2))*sizeof(float);
	size_t sharedSize_i = (size_t)(TX*TY*TZ)*sizeof(int);

	makeBufferKernel << <gridSize, blockSize, sharedSize >> >(d_in, d_vol, 'm', volSize);
	copyBufferKernel << <gridSize, blockSize >> >(buffer, d_vol, volSize);

	for (int dilate = 0; dilate < DNE; ++dilate)
	{
		dilateKernel << <gridSize, blockSize, sharedSize >> >(buffer, d_vol, volSize);
		copyBufferKernel << <gridSize, blockSize >> >(d_vol, buffer, volSize);
	}

	while (h_norm > STD*STD*(volSize.x*volSize.y*volSize.z - 1))
	{
		plusBufferKernel << <gridSize, blockSize, sharedSize >> >(buffer, d_vol, volSize);
		cudaMemset(d_norm, 0, sizeof(float));
		normBufferKernel << <gridSize, blockSize, sharedSize >> >(d_norm, d_vol, buffer, volSize);
		cudaMemcpy(&h_norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
		copyBufferKernel << <gridSize, blockSize >> >(d_vol, buffer, volSize);
	}

	h_norm = STD*STD*(volSize.x*volSize.y*volSize.z - 1) + 1;
	while (h_norm > STD*STD*(volSize.x*volSize.y*volSize.z - 1))
	{
		avgBufferKernel << <gridSize, blockSize, sharedSize >> >(buffer, d_vol, volSize);
		cudaMemset(d_norm, 0, sizeof(float));
		normBufferKernel << <gridSize, blockSize, sharedSize >> >(d_norm, d_vol, buffer, volSize);
		cudaMemcpy(&h_norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
		copyBufferKernel << <gridSize, blockSize >> >(d_vol, buffer, volSize);
	}

	h_norm = STD*STD*(volSize.x*volSize.y*volSize.z - 1) + 1;
	while (h_norm > STD*STD*(volSize.x*volSize.y*volSize.z - 1))
	{
		extremeBufferKernel << <gridSize, blockSize, sharedSize >> >(buffer, d_vol, volSize);
		cudaMemset(d_norm, 0, sizeof(float));
		normBufferKernel << <gridSize, blockSize, sharedSize >> >(d_norm, d_vol, buffer, volSize);
		cudaMemcpy(&h_norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
		copyBufferKernel << <gridSize, blockSize >> >(d_vol, buffer, volSize);
	}

	findExmKernel << <gridSize, blockSize, sharedSize_i >> >(d_max, d_min, d_vol, volSize);
	fillKernel << <gridSize, blockSize, sharedSize >> >(d_max, d_min, d_vol, buffer, volSize);

	for (int erose = 0; erose < DNE - 1; ++erose)
	{
		eroseKernel << <gridSize, blockSize, sharedSize >> >(buffer, d_vol, volSize);
		copyBufferKernel << <gridSize, blockSize >> >(d_vol, buffer, volSize);
	}

	findBondaryKernel << <gridSize, blockSize, sharedSize >> >(d_vol, buffer, volSize);

	h_norm = STD*STD*(volSize.x*volSize.y*volSize.z - 1) + 1;
	while (h_norm > STD*STD*(volSize.x*volSize.y*volSize.z - 1))
	{
		plusBufferKernel << <gridSize, blockSize, sharedSize >> >(buffer, d_vol, volSize);
		cudaMemset(d_norm, 0, sizeof(float));
		normBufferKernel << <gridSize, blockSize, sharedSize >> >(d_norm, d_vol, buffer, volSize);
		cudaMemcpy(&h_norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
		copyBufferKernel << <gridSize, blockSize >> >(d_vol, buffer, volSize);
	}

	mapBufferKernel << <gridSize, blockSize >> >(d_in, d_vol, 'm', volSize);
	//showBufferKernel << <gridSize, blockSize >> >(d_in, d_vol, volSize);

	cudaFree(buffer);
	cudaFree(d_norm);
	cudaFree(d_min);
	cudaFree(d_max);
}

void fatKernelLauncher(uchar4 *d_in, float *d_vol, int3 volSize)
{
	float *buffer = 0, *d_norm = 0;
	int *d_min = 0, *d_max = 0;
	int h_min = 0, h_max = 0;
	float h_norm = STD*STD*(volSize.x*volSize.y*volSize.z - 1) + 1;
	cudaMalloc(&buffer, volSize.x*volSize.y*volSize.z*sizeof(float));
	cudaMalloc(&d_norm, sizeof(float));
	cudaMalloc(&d_min, sizeof(int));
	cudaMalloc(&d_max, sizeof(int));
	cudaMemset(d_min, 0, sizeof(int));
	cudaMemset(d_max, 0, sizeof(int));
	dim3 blockSize(TX, TY, TZ);
	dim3 gridSize(divUp(volSize.x, TX), divUp(volSize.y, TY), divUp(volSize.z, TZ));
	size_t sharedSize = (size_t)((TX + 2)*(TY + 2)*(TZ + 2))*sizeof(float);

	makeBufferKernel << <gridSize, blockSize, sharedSize >> >(d_in, d_vol, 's', volSize);

	while (h_norm > STD*STD*(volSize.x*volSize.y*volSize.z - 1))
	{
		plusBufferKernel << <gridSize, blockSize, sharedSize >> >(buffer, d_vol, volSize);
		cudaMemset(d_norm, 0, sizeof(float));
		normBufferKernel << <gridSize, blockSize, sharedSize >> >(d_norm, d_vol, buffer, volSize);
		cudaMemcpy(&h_norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
		copyBufferKernel << <gridSize, blockSize >> >(d_vol, buffer, volSize);
	}

	findBondaryKernel << <gridSize, blockSize, sharedSize >> >(d_vol, buffer, volSize);

	while (h_norm > STD*STD*(volSize.x*volSize.y*volSize.z - 1))
	{
		plusBufferKernel << <gridSize, blockSize, sharedSize >> >(buffer, d_vol, volSize);
		cudaMemset(d_norm, 0, sizeof(float));
		normBufferKernel << <gridSize, blockSize, sharedSize >> >(d_norm, d_vol, buffer, volSize);
		cudaMemcpy(&h_norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
		copyBufferKernel << <gridSize, blockSize >> >(d_vol, buffer, volSize);
	}


	mapBufferKernel << <gridSize, blockSize >> >(d_in, d_vol, 's', volSize);
	//showBufferKernel << <gridSize, blockSize >> >(d_in, d_vol, volSize);

	cudaFree(buffer);
	cudaFree(d_norm);
	cudaFree(d_min);
	cudaFree(d_max);
}

void exportLauncher(uchar4 *d_in, int3 volSize)
{
	uchar4 *img = (uchar4*)malloc(volSize.x*volSize.y*volSize.z*sizeof(uchar4));
	cudaMemcpy(img, d_in, volSize.x*volSize.y*volSize.z*sizeof(uchar4), cudaMemcpyDeviceToHost);
	for (int s = 0; s < volSize.z; ++s)
	{
		char exportFile[24];
		cimg_library::CImg<unsigned char>imageOut(volSize.x, volSize.y, 1, 3);
		for (int r = 0; r < volSize.y; ++r)
		{
			for (int c = 0; c < volSize.x; ++c)
			{
				int i = h_flatten(c, r, s, volSize.x, volSize.y, volSize.z);
				imageOut(c, r, 0) = img[i].x;
				imageOut(c, r, 1) = img[i].y;
				imageOut(c, r, 2) = img[i].z;
			}
		}
		sprintf(exportFile, "CUDA_Arm/arm_%i.bmp", s + 1);
		imageOut.save_bmp(exportFile);
	}
	free(img);
}

void volumeKernelLauncher(float *d_vol, int3 volSize, float4 params)
{
	dim3 blockSize(TX, TY, TZ);
	dim3 gridSize(divUp(volSize.x, TX), divUp(volSize.y, TY), divUp(volSize.z, TZ));
	volumeKernel << <gridSize, blockSize >> >(d_vol, volSize, params);
}

void kernelLauncher(uchar4 *d_out, uchar4 *d_in, float *d_vol, float *b_vol, float *m_vol, float *f_vol, int w, int h, int3 volSize, int3 parSize, float zs, float alpha, float theta, float gamma, bool b_disp, bool m_disp, bool f_disp, float dist)
{
	dim3 blockSize(TX2, TY2);
	dim3 gridSize(divUp(w, TX2), divUp(h, TY2));
	renderFloatKernel << <gridSize, blockSize >> >(d_out, d_in, d_vol, b_vol, m_vol, f_vol, w, h, volSize, parSize, zs, gamma, theta, alpha, b_disp, m_disp, f_disp, dist);
}