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

int main() 
{
	cudaMalloc(&d_img, volSize.x*volSize.y*volSize.z*volSize.w*sizeof(Npp8u));
	cudaMalloc(&d_muscle, volSize.x*volSize.y*volSize.z*volSize.w*sizeof(Npp8u));
	cudaMalloc(&d_fat, volSize.x*volSize.y*volSize.z*volSize.w*sizeof(Npp8u));
	cudaMalloc(&d_skin, volSize.x*volSize.y*volSize.z*volSize.w*sizeof(Npp8u));
	cudaMalloc(&d_bone, volSize.x*volSize.y*volSize.z*volSize.w*sizeof(Npp8u));

	importNPP(d_img, imgSize, volSize);

	colorSeparateNPP(d_img, d_bone, d_muscle, d_fat, d_skin, volSize);
	boneNPP(d_bone, boneDandE, volSize);
	muscleNPP(d_muscle, muscleDandE, volSize);
	fatNPP(d_fat, blendDist, volSize);
	skinNPP(d_skin, d_fat, skinThickness, volSize);
	trimNPP(d_bone, d_muscle, d_fat, d_skin, volSize);

	imageAddNPP(d_img, d_bone, d_muscle, d_fat, d_skin, volSize);

	exportNPP(d_img, volSize);

	cudaFree(d_img);
	cudaFree(d_muscle);
	cudaFree(d_fat);
	cudaFree(d_skin);
	cudaFree(d_bone);
	return 0;
}