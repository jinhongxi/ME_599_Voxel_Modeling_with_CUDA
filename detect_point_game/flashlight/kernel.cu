#include "kernel.h"
#define TX 32
#define TY 32

__device__
unsigned char clip(int n)
{
	return n > 255 ? 255 : (n < 0 ? 0 : n);
}

__global__
void distanceKernel(uchar4 *d_out, int w, int h, int2 pos, int2 pos0)
{
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
	if ((c >= w) || (r >= h)) return;
	const int i = c + r * w;
	const int dist = sqrtf((c - pos.x)*(c - pos.x) + (r - pos.y)*(r - pos.y));
	const int dist0 = sqrtf((c - pos0.x)*(c - pos0.x) + (r - pos0.y)*(r - pos0.y));
	const unsigned char intensity = clip(255-dist);
	const unsigned char intensity0 = clip(255-dist0);
	const unsigned char red = clip(intensity*intensity0/255);
	const unsigned char blue = clip(intensity*(1-intensity0 / 255));
	d_out[i].x = red;
	d_out[i].y = 0;
	d_out[i].z = blue;
	d_out[i].w = 255;
}

void kernelLauncher(uchar4 *d_out, int w, int h, int2 pos, int2 pos0)
{
	const dim3 blockSize(TX, TY);
	const dim3 gridSize = dim3((w + TX - 1) / TX, (h + TY - 1) / TY);
	distanceKernel << <gridSize, blockSize >> >(d_out, w, h, pos, pos0);
}