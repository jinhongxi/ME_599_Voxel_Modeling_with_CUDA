#ifndef CUDAFUNC_CUH
#define CUDAFUNC_CUH

typedef struct {
	float3 o, d;
} Ray;

int divUp(int a, int Ta);

int h_idxClip(int idx, int idxMax);

int h_flatten(int c, int r, int s, int w, int h, int t);

uchar4 colorConvert(uchar4 in);

__device__
unsigned char clip(int n);

__device__
int idxClip(int idx, int idxMax);

__device__
int clipWithBounds(int n, int n_min, int n_max);

__device__
int flatten(int c, int r, int s, int w, int h, int t);

__device__
float ucharToFloat(uchar4 color, char channel);

__device__
uchar4 floatToUchar(uchar4 color, float n, char channel);

__device__
float3 xRotate(float3 pos, float theta);

__device__
float3 yRotate(float3 pos, float theta);

__device__
float3 zRotate(float3 pos, float theta);

__device__ 
float3 scrIdxToPos(int c, int r, int w, int h, float zs);

__device__ 
float3 paramRay(Ray r, float t);

__device__ 
float planeSDF(float3 pos, float3 norm, float d);

__device__
bool rayPlaneIntersect(Ray myRay, float3 n, float dist, float *t);

__device__
bool intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar);

__device__
int3 posToVolIndex(float3 pos, int3 volSize);

__device__
float func(int c, int r, int s, int3 volSize, float4 params);

__device__
uchar4 rayCastShader(uchar4 *d_in, float *d_vol, float *b_vol, float *m_vol, float *f_vol, int3 volSize, int3 parSize, Ray boxRay, bool b_disp, bool m_disp, bool f_disp, float dist);

__global__
void showBufferKernel(uchar4 *img, float *buf, int3 volSize);

__global__
void findExmKernel(int *d_max, int *d_min, float *d_vol, int3 volSize);

__global__
void makeBufferKernel(uchar4 *img, float *buf, char channel, int3 volSize);

__global__
void mapBufferKernel(uchar4 *img, float *buf, char channel, int3 volSize);

__global__
void plusBufferKernel(float *buf2, float *buf1, int3 volSize);

__global__
void copyBufferKernel(float *buf2, float *buf1, int3 volSize);

__global__
void normBufferKernel(float *norm, float *buf2, float *buf1, int3 volSize);

__global__
void surfaceKernel(int *area, float *buf, int3 volSize);

__global__
void extremeBufferKernel(float *buf2, float *buf1, int3 volSize);

__global__
void avgBufferKernel(float *buf2, float *buf1, int3 volSize);

__global__
void fillKernel(int *d_max, int *d_min, float *buf2, float *buf1, int3 volSize);

__global__
void dilateKernel(float *buf2, float *buf1, int3 volSize);

__global__
void eroseKernel(float *buf2, float *buf1, int3 volSize);

__global__
void boneCleanKernel(float *buf2, float *buf1, int3 volSize);

__global__
void findBondaryKernel(float *buf2, float *buf1, int3 volSize);

__global__
void volumeKernel(float *d_vol, int3 volSize, float4 params);

__global__
void renderFloatKernel(uchar4 *d_out, uchar4 *d_in, float *d_vol, float *b_vol, float *m_vol, float *f_vol, int w, int h, int3 volSize, int3 parSize, float zs, float gamma, float theta, float alpha, bool b_disp, bool m_disp, bool f_disp, float dist);

#endif