#ifndef CUDAFUNC_CUH
#define CUDAFUNC_CUH

#include <math.h>
#include <helper_math.h>
#include <npp.h>

typedef struct {
	float3 o, d;
} Ray;

int divUp(int a, int A);

__device__
int flatten(int ch, int3 i, int4 volSize);

__device__
int clipWithBounds(int n, int n_min, int n_max);

__device__
float3 xRotate(float3 pos, float theta);

__device__
float3 yRotate(float3 pos, float theta);

__device__
float3 zRotate(float3 pos, float theta);

__device__
float3 scrIdxToPos(int c, int r, int w, int h, float zs);

__device__
int3 posToVolIndex(float3 pos, int4 volSize);

__device__
float3 paramRay(Ray r, float t);

__device__
bool intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar);

__device__
uchar4 rayCastShader(Npp8u *d_bound, int4 volSize, Ray boxRay, float dist);

__global__
void renderFloatKernel(uchar4 *d_out, Npp8u *d_bound, int w, int h, int4 volSize, float alpha, float theta, float gamma, float dist);


#endif