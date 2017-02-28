#include <stdio.h>
#include <stdlib.h>
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

#include "kernel.h"
#include "interactions.h"

int main()
{
	cudaMalloc(&b_vol, volSize.x*volSize.y*volSize.z*sizeof(float));
	cudaMalloc(&m_vol, volSize.x*volSize.y*volSize.z*sizeof(float));
	cudaMalloc(&f_vol, volSize.x*volSize.y*volSize.z*sizeof(float));
	cudaMalloc(&d_in, volSize.x*volSize.y*volSize.z*sizeof(uchar4));

	printf("  Loading...\n");
	importLauncher(d_in, volSize);

	printf("  Buffering...\n");
	boneKernelLauncher(d_in, b_vol, volSize);
	printf("  Bone completed...\n");
	muscleKernelLauncher(d_in, m_vol, volSize);
	printf("  Muscle completed...\n");
	fatKernelLauncher(d_in, f_vol, volSize);
	printf("  Fat completed...\n");

	printf("  Printing...\n");
	exportLauncher(d_in, volSize);

	cudaFree(b_vol);
	cudaFree(m_vol);
	cudaFree(f_vol);
	cudaFree(d_in);
	return 0;
}