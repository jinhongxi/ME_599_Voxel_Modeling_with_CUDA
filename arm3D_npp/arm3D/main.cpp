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

GLuint pbo = 0;
GLuint tex = 0;
struct cudaGraphicsResource *cuda_pbo_resource;

void render()
{
	uchar4 *d_out = 0;
	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL, cuda_pbo_resource);
	kernelLauncher(d_out, d_bound, W, H, volSize, alpha, theta, gamma, dist);
	if (print)
	{
		exportNPP(d_img, volSize);
		printf("\n    Updated. \n");
		print = false;
	}
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
	char title[128];
	sprintf(title, "Arm Segmentation : dist = %.1f, x = %.1f, y = %.1f, z = %.1f", dist, alpha, theta, gamma);
	glutSetWindowTitle(title);
}

void draw_texture()
{
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0, H);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(W, H);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(W, 0);
	glEnd();
	glDisable(GL_TEXTURE_2D);
}

void display()
{
	render();
	draw_texture();
	glutSwapBuffers();
}

void initGLUT(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(W, H);
	glutCreateWindow(TITLE_STRING);
#ifndef __APPLE__
	glewInit();
#endif
}

void initPixelBuffer()
{
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, W*H*sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

void exitfunc()
{
	if (pbo)
	{
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &pbo);
		glDeleteTextures(1, &tex);
	}

	cudaFree(d_img);
	cudaFree(d_muscle);
	cudaFree(d_fat);
	cudaFree(d_skin);
	cudaFree(d_bone);
	cudaFree(d_bound);
	cudaFree(d_origin);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceReset();
}

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
	cudaMalloc(&d_origin, volSize.x*volSize.y*volSize.z*volSize.w*sizeof(Npp8u));

	importNPP(d_img, imgSize, volSize);
	copyNPP(d_img, d_origin, volSize);
	colorSeparateNPP(d_img, d_bone, d_muscle, d_fat, d_skin, volSize);

	nppLauncher(d_img, d_bone, d_muscle, d_fat, d_skin, boneDandE, muscleDandE, blendDist, skinThickness, soften, volSize);
	boundaryLauncher(d_bound, d_origin, d_img, d_bone, d_muscle, d_fat, d_skin, volSize, showBone, showMuscle, showFat, showSkin, showDiff, showOrig);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float cudaTime = 0.f;
	cudaEventElapsedTime(&cudaTime, start, stop);

	printInstructions();
	printf("\n    (CUDA Time = %f Ms)\n", cudaTime);
	initGLUT(&argc, argv);
	createMenu();
	gluOrtho2D(0, W, H, 0);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(handleSpecialKeypress);
	glutDisplayFunc(display);
	initPixelBuffer();
	glutMainLoop();
	atexit(exitfunc);

	return 0;
}