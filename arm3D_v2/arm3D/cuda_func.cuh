#ifndef CUDAFUNC_CUH
#define CUDAFUNC_CUH

int divUp(int a, int Ta);

uchar4 colorConvert(uchar4 n);

int h_idxClip(int idx, int idxMax);

int h_flatten(int c, int r, int s, int w, int h, int t);

__device__
unsigned char clip(int n);

__device__
int idxClip(int idx, int idxMax);

__device__
int flatten(int c, int r, int s, int w, int h, int t);

__device__
float ucharToFloat(uchar4 color4, char channel);

__device__
uchar4 floatToUchar(uchar4 color, float n, char channel);

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
void normBufferKernel(float *norm, float *buf2, float *buf1, int3 volSize);

__global__
void surfaceKernel(int *area, float *buf, int3 volSize);

__global__
void avgBufferKernel(float *buf2, float *buf1, int3 volSize);

__global__
void extremeBufferKernel(float *buf2, float *buf1, int3 volSize);

#endif