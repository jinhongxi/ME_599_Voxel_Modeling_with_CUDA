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

	importKernel(d_img, imgSize, volSize);

	exportKernel(d_img, volSize);

	cudaFree(d_img);
	return 0;
}