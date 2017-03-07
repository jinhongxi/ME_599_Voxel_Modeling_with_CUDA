#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <stdlib.h>
#include <npp.h>

#include "kernel.h"
#include "interactions.h"

int main(int argc, char** argv)
{
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	cudaMalloc(&d_img, volSize.x*volSize.y*volSize.z*volSize.w*sizeof(Npp8u));
	cudaMalloc(&d_muscle, volSize.x*volSize.y*volSize.z*volSize.w*sizeof(Npp8u));
	cudaMalloc(&d_fat, volSize.x*volSize.y*volSize.z*volSize.w*sizeof(Npp8u));
	cudaMalloc(&d_skin, volSize.x*volSize.y*volSize.z*volSize.w*sizeof(Npp8u));
	cudaMalloc(&d_bone, volSize.x*volSize.y*volSize.z*volSize.w*sizeof(Npp8u));
	cudaMalloc(&d_bound, volSize.x*volSize.y*volSize.z*volSize.w*sizeof(Npp8u));
	importNPP(d_img, imgSize, volSize);
	colorSeparateNPP(d_img, d_bone, d_muscle, d_fat, d_skin, volSize);

	nppLauncher(d_img, d_bone, d_muscle, d_fat, d_skin, boneDandE, muscleDandE, blendDist, skinThickness, volSize);
	boundaryLauncher(d_bound, d_bone, d_muscle, d_fat, d_skin, volSize, showBone, showMuscle, showFat, showSkin);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float cudaTime = 0.f;
	cudaEventElapsedTime(&cudaTime, start, stop);
	printf("\n    (CUDA Time = %f Ms)\n", cudaTime);

	if (print)
	{
		exportNPP(d_img, volSize);
		print = false;
	}

	cudaFree(d_img);
	cudaFree(d_muscle);
	cudaFree(d_fat);
	cudaFree(d_skin);
	cudaFree(d_bone);
	cudaFree(d_bound);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceReset();

	return 0;
}