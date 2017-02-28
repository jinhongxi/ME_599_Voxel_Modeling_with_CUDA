#include "cuda_func.cuh"
#include <math.h>
#include <helper_math.h>

#define EPS 0.01f
#define THRESHOLD 0.95

int divUp(int a, int Ta)
{
	return (a + Ta - 1) / Ta;
}

int h_idxClip(int idx, int idxMax)
{
	return idx >(idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

int h_flatten(int c, int r, int s, int w, int h, int t)
{
	return h_idxClip(c, w) + w*h_idxClip(r, h) + w*h*h_idxClip(s, t);
}

uchar4 colorConvert(uchar4 in)
{
	uchar4 out = { 0, 0, 0, 0 };
	if (in.x > 150 && in.y > 150 && in.z > 150) out = { 255, 255, 255, 255 };
	else if (in.x > 150 && in.y > 150 && in.z <= 150) out = { 0, 0, 255, 255 };
	else if (in.x > 150 && in.y <= 150 && in.z <= 150) out = { 255, 0, 0, 255 };
	return out;
}

__device__
unsigned char clip(int n)
{
	return n > 255 ? 255 : (n < 0 ? 0 : n);
}

__device__
int idxClip(int idx, int idxMax)
{
	return idx >(idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

__device__
int flatten(int c, int r, int s, int w, int h, int t)
{
	return idxClip(c, w) + w*idxClip(r, h) + w*h*idxClip(s, t);
}

__device__
float ucharToFloat(uchar4 color,char channel)
{
	switch (channel)
	{
	case 'm':
	{
		if (color.x > 150 && color.y <= 150 && color.z <= 150) return -1.f;
		else if (color.x > 150 && color.y > 150 && color.z > 150) return -1.f;
		else return 0.f;
		break;
	}
	case 'f':
	{
		if (color.x <= 150 && color.y <= 150 && color.z > 150) return -1.f;
		else return 0.f;
		break;
	}
	case 's':
	{
		if (color.x > 150 || color.y > 150 || color.z > 150) return -1.f;
		else return 0.f;
		break;
	}
	case 'b': 
	{
		if (color.x > 150 && color.y > 150 && color.z > 150) return -1.f;
		else return 0.f;
		break;
	}
	default: return 0.f;
	}
}

__device__
uchar4 floatToUchar(uchar4 color, float n, char channel)
{
	uchar4 change = color;
	switch (channel)
	{
	case 'm':
	{
		if (color.x > 150 && color.y > 150 && color.z > 150) change = { 255, 255, 255, 255 };
		else if (n <= 0.f) change = { 255, 0, 0, 255 };
		else if (color.x > 150) change = { 0, 0, 255, 255 };
		break;
	}
	case 's':
	{
		if (color.x > 150 && color.y > 150 && color.z > 150) change = { 255, 255, 255, 255 };
		else if (color.x > 150) change = { 255, 0, 0, 255 };
		else if (abs(n) < EPS) change = { 0, 255, 0, 255 };
		else if (n <= 0.f) change = { 0, 255 - clip(-255 * n), clip(-255 * n), 255 };
		else change = { 0, 0, 0, 255 };
		break;
	}
	case 'f':
	{
		if (n <= 0.f) change = { 0, 255 - clip(-100 * n), clip(-100 * n), 255 };
		break;
	}
	case 'b':
	{
		if (color.x > 150 && color.y > 150 && color.z > 150 && n >= 1.f) change = { 255, 0, 0, 255 };
		else if (n <= 0.f) change = { 255, 255, 255, 255 };
		break;
	}
	default: change = color;
	}
	return change;
}

__global__
void showBufferKernel(uchar4 *img, float *buf, int3 volSize)
{
	const int c = threadIdx.x + blockDim.x*blockIdx.x;
	const int r = threadIdx.y + blockDim.y*blockIdx.y;
	const int s = threadIdx.z + blockDim.z*blockIdx.z;
	if (c >= volSize.x || r >= volSize.y || s >= volSize.z) return;
	const int i = flatten(c, r, s, volSize.x, volSize.y, volSize.z);

	if (abs(buf[i]) < EPS) img[i] = { 255, 255, 255, 255 };
	else if (buf[i] > 0)
	{
		img[i].x = 0;
		img[i].y = 0;
		img[i].z = 255 - clip(20 * buf[i]);
		img[i].w = 0;
	}
	else
	{
		img[i].x = 255;
		img[i].y = 255 - clip(-20 * buf[i]);
		img[i].z = 255;
		img[i].w = 0;
	}
}

__global__
void findExmKernel(int *d_max, int *d_min, float *d_vol, int3 volSize)
{
	const int c = threadIdx.x + blockDim.x*blockIdx.x;
	const int r = threadIdx.y + blockDim.y*blockIdx.y;
	const int s = threadIdx.z + blockDim.z*blockIdx.z;
	const int i = flatten(c, r, s, volSize.x, volSize.y, volSize.z);

	const int s_c = threadIdx.x;
	const int s_r = threadIdx.y;
	const int s_s = threadIdx.z;
	const int s_w = blockDim.x;
	const int s_h = blockDim.y;
	const int s_t = blockDim.z;
	const int s_i = flatten(s_c, s_r, s_s, s_w, s_h, s_t);

	extern __shared__ int s_int[];

	if (c >= volSize.x || r >= volSize.y || s >= volSize.z)
	{
		s_int[s_i] = 0;
		return;
	}
	else s_int[s_i] = (int)d_vol[i];
	__syncthreads();

	if (s_i == 0)
	{
		int min0 = 0, max0 = 0;
		for (int z = 0; z < s_t; ++z)
		{
			for (int y = 0; y < s_h; ++y)
			{
				for (int x = 0; x < s_w; ++x)
				{
					min0 = fminf(min0, s_int[flatten(x, y, z, s_w, s_h, s_t)]);
					max0 = fmaxf(max0, s_int[flatten(x, y, z, s_w, s_h, s_t)]);
				}
			}
		}
		atomicMin(d_min, min0);
		atomicMax(d_max, max0);
	}
}

__global__
void makeBufferKernel(uchar4 *img, float *buf, char channel, int3 volSize)
{
	const int c = threadIdx.x + blockDim.x*blockIdx.x;
	const int r = threadIdx.y + blockDim.y*blockIdx.y;
	const int s = threadIdx.z + blockDim.z*blockIdx.z;
	if (c >= volSize.x || r >= volSize.y || s >= volSize.z) return;
	const int i = flatten(c, r, s, volSize.x, volSize.y, volSize.z);

	const int s_c = threadIdx.x + 1;
	const int s_r = threadIdx.y + 1;
	const int s_s = threadIdx.z + 1;
	const int s_w = blockDim.x + 2;
	const int s_h = blockDim.y + 2;
	const int s_t = blockDim.z + 2;
	const int s_i = flatten(s_c, s_r, s_s, s_w, s_h, s_t);

	extern __shared__ float s_float[];

	s_float[s_i] = ucharToFloat(img[i], channel);

	if (threadIdx.x < 1)
	{
		s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] = ucharToFloat(img[flatten(c - 1, r, s, volSize.x, volSize.y, volSize.z)], channel);
		s_float[flatten(s_c + blockDim.x, s_r, s_s, s_w, s_h, s_t)] = ucharToFloat(img[flatten(c + blockDim.x, r, s, volSize.x, volSize.y, volSize.z)], channel);
	}
	if (threadIdx.y < 1)
	{
		s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] = ucharToFloat(img[flatten(c, r - 1, s, volSize.x, volSize.y, volSize.z)], channel);
		s_float[flatten(s_c, s_r + blockDim.y, s_s, s_w, s_h, s_t)] = ucharToFloat(img[flatten(c, r + blockDim.y, s, volSize.x, volSize.y, volSize.z)], channel);
	}
	if (threadIdx.z < 1)
	{
		s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)] = ucharToFloat(img[flatten(c, r, s - 1, volSize.x, volSize.y, volSize.z)], channel);
		s_float[flatten(s_c, s_r, s_s + blockDim.z, s_w, s_h, s_t)] = ucharToFloat(img[flatten(c, r, s + blockDim.z, volSize.x, volSize.y, volSize.z)], channel);
	}
	__syncthreads();

	if (s_float[s_i] <= -1.f) buf[i] = -1.f;
	else buf[i] = 1.f;
}

__global__
void mapBufferKernel(uchar4 *img, float *buf, char channel, int3 volSize)
{
	const int c = threadIdx.x + blockDim.x*blockIdx.x;
	const int r = threadIdx.y + blockDim.y*blockIdx.y;
	const int s = threadIdx.z + blockDim.z*blockIdx.z;
	if (c >= volSize.x || r >= volSize.y || s >= volSize.z) return;
	const int i = flatten(c, r, s, volSize.x, volSize.y, volSize.z);

	img[i] = floatToUchar(img[i], buf[i], channel);
}

__global__
void plusBufferKernel(float *buf2, float *buf1, int3 volSize)
{
	const int c = threadIdx.x + blockDim.x*blockIdx.x;
	const int r = threadIdx.y + blockDim.y*blockIdx.y;
	const int s = threadIdx.z + blockDim.z*blockIdx.z;
	if (c >= volSize.x || r >= volSize.y || s >= volSize.z) return;
	const int i = flatten(c, r, s, volSize.x, volSize.y, volSize.z);

	const int s_c = threadIdx.x + 1;
	const int s_r = threadIdx.y + 1;
	const int s_s = threadIdx.z + 1;
	const int s_w = blockDim.x + 2;
	const int s_h = blockDim.y + 2;
	const int s_t = blockDim.z + 2;
	const int s_i = flatten(s_c, s_r, s_s, s_w, s_h, s_t);

	extern __shared__ float s_float[];

	s_float[s_i] = buf1[i];

	if (threadIdx.x < 1)
	{
		s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] = buf1[flatten(c - 1, r, s, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c + blockDim.x, s_r, s_s, s_w, s_h, s_t)] = buf1[flatten(c + blockDim.x, r, s, volSize.x, volSize.y, volSize.z)];
	}
	if (threadIdx.y < 1)
	{
		s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] = buf1[flatten(c, r - 1, s, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c, s_r + blockDim.y, s_s, s_w, s_h, s_t)] = buf1[flatten(c, r + blockDim.y, s, volSize.x, volSize.y, volSize.z)];
	}
	if (threadIdx.z < 1)
	{
		s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)] = buf1[flatten(c, r, s - 1, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c, s_r, s_s + blockDim.z, s_w, s_h, s_t)] = buf1[flatten(c, r, s + blockDim.z, volSize.x, volSize.y, volSize.z)];
	}
	__syncthreads();

	if (s_float[s_i] < 0.f)
	{
		float p = 0.f;
		if (abs(s_float[s_i] - s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)]) < EPS && abs(s_float[s_i] - s_float[flatten(s_c + 1, s_r, s_s, s_w, s_h, s_t)]) < EPS) p++;
		if (abs(s_float[s_i] - s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)]) < EPS && abs(s_float[s_i] - s_float[flatten(s_c, s_r + 1, s_s, s_w, s_h, s_t)]) < EPS) p++;
		if (s == volSize.z - 1) { if (abs(s_float[s_i] - s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)]) < EPS) p++; }
		else if (s == 0) { if (abs(s_float[s_i] - s_float[flatten(s_c, s_r, s_s + 1, s_w, s_h, s_t)]) < EPS) p++; }
		else { if (abs(s_float[s_i] - s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)]) < EPS && abs(s_float[s_i] - s_float[flatten(s_c, s_r, s_s + 1, s_w, s_h, s_t)]) < EPS) p++; }
		buf2[i] = -sqrtf(s_float[s_i] * s_float[s_i] + p);
	}
	else if (s_float[s_i] > 0.f)
	{
		float p = 0.f;
		if (c == volSize.x - 1) { if (abs(s_float[s_i] - s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)]) < EPS) p++; }
		else if (c == 0) { if (abs(s_float[s_i] - s_float[flatten(s_c + 1, s_r, s_s, s_w, s_h, s_t)]) < EPS) p++; }
		else { if (abs(s_float[s_i] - s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)]) < EPS && abs(s_float[s_i] - s_float[flatten(s_c + 1, s_r, s_s, s_w, s_h, s_t)]) < EPS) p++; }
		if (r == volSize.y - 1) { if (abs(s_float[s_i] - s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)]) < EPS) p++; }
		else if (r == 0) { if (abs(s_float[s_i] - s_float[flatten(s_c, s_r + 1, s_s, s_w, s_h, s_t)]) < EPS) p++; }
		else { if (abs(s_float[s_i] - s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)]) < EPS && abs(s_float[s_i] - s_float[flatten(s_c, s_r + 1, s_s, s_w, s_h, s_t)]) < EPS) p++; }
		if (s == volSize.z - 1) { if (abs(s_float[s_i] - s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)]) < EPS) p++; }
		else if (s == 0) { if (abs(s_float[s_i] - s_float[flatten(s_c, s_r, s_s + 1, s_w, s_h, s_t)]) < EPS) p++; }
		else { if (abs(s_float[s_i] - s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)]) < EPS && abs(s_float[s_i] - s_float[flatten(s_c, s_r, s_s + 1, s_w, s_h, s_t)]) < EPS) p++; }
		buf2[i] = sqrtf(s_float[s_i] * s_float[s_i] + p);
	}
	else buf2[i] = s_float[s_i];
}

__global__
void copyBufferKernel(float *buf2, float *buf1, int3 volSize)
{
	const int c = threadIdx.x + blockDim.x*blockIdx.x;
	const int r = threadIdx.y + blockDim.y*blockIdx.y;
	const int s = threadIdx.z + blockDim.z*blockIdx.z;
	if (c >= volSize.x || r >= volSize.y || s >= volSize.z) return;
	const int i = flatten(c, r, s, volSize.x, volSize.y, volSize.z);

	buf2[i] = buf1[i];
}

__global__
void normBufferKernel(float *norm, float *buf2, float *buf1, int3 volSize)
{
	const int c = threadIdx.x + blockDim.x*blockIdx.x;
	const int r = threadIdx.y + blockDim.y*blockIdx.y;
	const int s = threadIdx.z + blockDim.z*blockIdx.z;
	const int i = flatten(c, r, s, volSize.x, volSize.y, volSize.z);

	const int s_c = threadIdx.x;
	const int s_r = threadIdx.y;
	const int s_s = threadIdx.z;
	const int s_w = blockDim.x;
	const int s_h = blockDim.y;
	const int s_t = blockDim.z;
	const int s_i = flatten(s_c, s_r, s_s, s_w, s_h, s_t);

	extern __shared__ float s_float[];

	if (c >= volSize.x || r >= volSize.y || s >= volSize.z)
	{
		s_float[s_i] = 0.f;
		return;
	}
	else s_float[s_i] = (buf2[i] - buf1[i])*(buf2[i] - buf1[i]);
	__syncthreads();

	if (s_i == 0)
	{
		float sum = 0;
		for (int z = 0; z < s_t; ++z)
		{
			for (int y = 0; y < s_h; ++y)
			{
				for (int x = 0; x < s_w; ++x)
				{
					sum += s_float[flatten(x, y, z, s_w, s_h, s_t)];
				}
			}
		}
		atomicAdd(norm, sum);
	}
}

__global__
void surfaceKernel(int *area, float *buf, int3 volSize)
{
	const int c = threadIdx.x + blockDim.x*blockIdx.x;
	const int r = threadIdx.y + blockDim.y*blockIdx.y;
	const int s = threadIdx.z + blockDim.z*blockIdx.z;
	const int i = flatten(c, r, s, volSize.x, volSize.y, volSize.z);

	const int s_c = threadIdx.x;
	const int s_r = threadIdx.y;
	const int s_s = threadIdx.z;
	const int s_w = blockDim.x;
	const int s_h = blockDim.y;
	const int s_t = blockDim.z;
	const int s_i = flatten(s_c, s_r, s_s, s_w, s_h, s_t);

	extern __shared__ int s_int[];

	if (c >= volSize.x || r >= volSize.y || s >= volSize.z)
	{
		s_int[s_i] = 0;
		return;
	}
	else
	{
		if (buf[i] == 0.f) s_int[s_i] = 1;
		else s_int[s_i] = 0;
	}
	__syncthreads();

	if (s_i == 0)
	{
		float sum = 0;
		for (int z = 0; z < s_t; ++z)
		{
			for (int y = 0; y < s_h; ++y)
			{
				for (int x = 0; x < s_w; ++x)
				{
					sum += s_int[flatten(x, y, z, s_w, s_h, s_t)];
				}
			}
		}
		atomicAdd(area, sum);
	}
}

__global__
void extremeBufferKernel(float *buf2, float *buf1, int3 volSize)
{
	const int c = threadIdx.x + blockDim.x*blockIdx.x;
	const int r = threadIdx.y + blockDim.y*blockIdx.y;
	const int s = threadIdx.z + blockDim.z*blockIdx.z;
	if (c >= volSize.x || r >= volSize.y || s >= volSize.z) return;
	const int i = flatten(c, r, s, volSize.x, volSize.y, volSize.z);

	const int s_c = threadIdx.x + 1;
	const int s_r = threadIdx.y + 1;
	const int s_s = threadIdx.z + 1;
	const int s_w = blockDim.x + 2;
	const int s_h = blockDim.y + 2;
	const int s_t = blockDim.z + 2;
	const int s_i = flatten(s_c, s_r, s_s, s_w, s_h, s_t);

	extern __shared__ float s_float[];

	s_float[s_i] = buf1[i];

	if (threadIdx.x < 1)
	{
		s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] = buf1[flatten(c - 1, r, s, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c + blockDim.x, s_r, s_s, s_w, s_h, s_t)] = buf1[flatten(c + blockDim.x, r, s, volSize.x, volSize.y, volSize.z)];
	}
	if (threadIdx.y < 1)
	{
		s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] = buf1[flatten(c, r - 1, s, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c, s_r + blockDim.y, s_s, s_w, s_h, s_t)] = buf1[flatten(c, r + blockDim.y, s, volSize.x, volSize.y, volSize.z)];
	}
	if (threadIdx.z < 1)
	{
		s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)] = buf1[flatten(c, r, s - 1, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c, s_r, s_s + blockDim.z, s_w, s_h, s_t)] = buf1[flatten(c, r, s + blockDim.z, volSize.x, volSize.y, volSize.z)];
	}

	__syncthreads();

	if (s_float[s_i] < 0.f)
	{
		float min1 = 0.f, min2 = 0.f, min3 = 0.f, min0 = s_float[s_i];
		min1 = fminf(s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)], s_float[flatten(s_c + 1, s_r, s_s, s_w, s_h, s_t)]);
		min2 = fminf(s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)], s_float[flatten(s_c, s_r + 1, s_s, s_w, s_h, s_t)]);
		if (s == 0) min3 = s_float[flatten(s_c, s_r, s_s + 1, s_w, s_h, s_t)];
		else if (s == volSize.z - 1) min3 = s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)];
		else min3 = fminf(s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)], s_float[flatten(s_c, s_r, s_s + 1, s_w, s_h, s_t)]);
		min0 = fminf(min0, min1);
		min0 = fminf(min0, min2);
		min0 = fminf(min0, min3);
		buf2[i] = min0;
	}
	else if (s_float[s_i] > 0.f)
	{
		float max1 = 0.f, max2 = 0.f, max3 = 0.f, max0 = s_float[s_i];
		if (c == 0) max1 = s_float[flatten(s_c + 1, s_r, s_s, s_w, s_h, s_t)];
		else if (c == volSize.x - 1) max1 = s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)];
		else max1 = fmaxf(s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)], s_float[flatten(s_c + 1, s_r, s_s, s_w, s_h, s_t)]);
		if (r == 0) max2 = s_float[flatten(s_c, s_r + 1, s_s, s_w, s_h, s_t)];
		else if (r == volSize.y - 1) max2 = s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)];
		else max2 = fmaxf(s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)], s_float[flatten(s_c, s_r + 1, s_s, s_w, s_h, s_t)]);
		if (s == 0) max3 = s_float[flatten(s_c, s_r, s_s + 1, s_w, s_h, s_t)];
		else if (s == volSize.z - 1) max3 = s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)];
		else max3 = fmaxf(s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)], s_float[flatten(s_c, s_r, s_s + 1, s_w, s_h, s_t)]);
		max0 = fmaxf(max0, max1);
		max0 = fmaxf(max0, max2);
		max0 = fmaxf(max0, max3);
		buf2[i] = max0;
	}
	else buf2[i] = s_float[s_i];
}

__global__
void avgBufferKernel(float *buf2, float *buf1, int3 volSize)
{
	const int c = threadIdx.x + blockDim.x*blockIdx.x;
	const int r = threadIdx.y + blockDim.y*blockIdx.y;
	const int s = threadIdx.z + blockDim.z*blockIdx.z;
	if (c >= volSize.x || r >= volSize.y || s >= volSize.z) return;
	const int i = flatten(c, r, s, volSize.x, volSize.y, volSize.z);

	const int s_c = threadIdx.x + 1;
	const int s_r = threadIdx.y + 1;
	const int s_s = threadIdx.z + 1;
	const int s_w = blockDim.x + 2;
	const int s_h = blockDim.y + 2;
	const int s_t = blockDim.z + 2;
	const int s_i = flatten(s_c, s_r, s_s, s_w, s_h, s_t);

	extern __shared__ float s_float[];

	s_float[s_i] = buf1[i];

	if (threadIdx.x < 1)
	{
		s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] = buf1[flatten(c - 1, r, s, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c + blockDim.x, s_r, s_s, s_w, s_h, s_t)] = buf1[flatten(c + blockDim.x, r, s, volSize.x, volSize.y, volSize.z)];
	}
	if (threadIdx.y < 1)
	{
		s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] = buf1[flatten(c, r - 1, s, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c, s_r + blockDim.y, s_s, s_w, s_h, s_t)] = buf1[flatten(c, r + blockDim.y, s, volSize.x, volSize.y, volSize.z)];
	}
	if (threadIdx.z < 1)
	{
		s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)] = buf1[flatten(c, r, s - 1, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c, s_r, s_s + blockDim.z, s_w, s_h, s_t)] = buf1[flatten(c, r, s + blockDim.z, volSize.x, volSize.y, volSize.z)];
	}
	__syncthreads();

	if (s_float[s_i] < 0.f)
	{
		float p = s_float[s_i];
		int i = 1;
		if (s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] < 0.f) { p += s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)]; i++; }
		if (s_float[flatten(s_c + 1, s_r, s_s, s_w, s_h, s_t)] < 0.f) { p += s_float[flatten(s_c + 1, s_r, s_s, s_w, s_h, s_t)]; i++; }
		if (s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] < 0.f) { p += s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)]; i++; }
		if (s_float[flatten(s_c, s_r + 1, s_s, s_w, s_h, s_t)] < 0.f) { p += s_float[flatten(s_c, s_r + 1, s_s, s_w, s_h, s_t)]; i++; }
		if (s > 0) { if (s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)] < 0.f) { p += s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)]; i++; } }
		if (s < volSize.z - 1) { if (s_float[flatten(s_c, s_r, s_s + 1, s_w, s_h, s_t)] < 0.f) { p += s_float[flatten(s_c, s_r, s_s + 1, s_w, s_h, s_t)]; i++; } }
		buf2[i] = p / (float)i;
	}
	else if (s_float[s_i] > 0.f)
	{
		float p = s_float[s_i];
		int i = 1;
		if (c > 0) { if (s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] > 0.f) { p += s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)]; i++; } }
		if (c < volSize.x) { if (s_float[flatten(s_c + 1, s_r, s_s, s_w, s_h, s_t)] > 0.f) { p += s_float[flatten(s_c + 1, s_r, s_s, s_w, s_h, s_t)]; i++; } }
		if (r > 0){ if (s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] > 0.f) { p += s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)]; i++; } }
		if (r < volSize.y) { if (s_float[flatten(s_c, s_r + 1, s_s, s_w, s_h, s_t)] > 0.f) { p += s_float[flatten(s_c, s_r + 1, s_s, s_w, s_h, s_t)]; i++; } }
		if (s > 0) { if (s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)] > 0.f) { p += s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)]; i++; } }
		if (s < volSize.z - 1) { if (s_float[flatten(s_c, s_r, s_s + 1, s_w, s_h, s_t)] > 0.f) { p += s_float[flatten(s_c, s_r, s_s + 1, s_w, s_h, s_t)]; i++; } }
		buf2[i] = p / (float)i;
	}
	else buf2[i] = s_float[s_i];
}

__global__
void fillKernel(int *d_max, int *d_min, float *buf2, float *buf1, int3 volSize)
{
	const int c = threadIdx.x + blockDim.x*blockIdx.x;
	const int r = threadIdx.y + blockDim.y*blockIdx.y;
	const int s = threadIdx.z + blockDim.z*blockIdx.z;
	if (c >= volSize.x || r >= volSize.y || s >= volSize.z) return;
	const int i = flatten(c, r, s, volSize.x, volSize.y, volSize.z);

	const int s_c = threadIdx.x + 1;
	const int s_r = threadIdx.y + 1;
	const int s_s = threadIdx.z + 1;
	const int s_w = blockDim.x + 2;
	const int s_h = blockDim.y + 2;
	const int s_t = blockDim.z + 2;
	const int s_i = flatten(s_c, s_r, s_s, s_w, s_h, s_t);

	extern __shared__ float s_float[];

	s_float[s_i] = buf1[i];

	if (threadIdx.x < 1)
	{
		s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] = buf1[flatten(c - 1, r, s, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c + blockDim.x, s_r, s_s, s_w, s_h, s_t)] = buf1[flatten(c + blockDim.x, r, s, volSize.x, volSize.y, volSize.z)];
	}
	if (threadIdx.y < 1)
	{
		s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] = buf1[flatten(c, r - 1, s, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c, s_r + blockDim.y, s_s, s_w, s_h, s_t)] = buf1[flatten(c, r + blockDim.y, s, volSize.x, volSize.y, volSize.z)];
	}
	if (threadIdx.z < 1)
	{
		s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)] = buf1[flatten(c, r, s - 1, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c, s_r, s_s + blockDim.z, s_w, s_h, s_t)] = buf1[flatten(c, r, s + blockDim.z, volSize.x, volSize.y, volSize.z)];
	}
	__syncthreads();

	if (s_float[s_i] > 0.f)
	{
		if (s_float[s_i] <= *d_max*THRESHOLD) buf2[i] = -1.f;
		else buf2[i] = 1.f;
	}
	else
	{
		if (s_float[s_i] >= *d_min*THRESHOLD) buf2[i] = 1.f;
		else buf2[i] = -1.f;
	}
}

__global__
void dilateKernel(float *buf2, float *buf1, int3 volSize)
{
	const int c = threadIdx.x + blockDim.x*blockIdx.x;
	const int r = threadIdx.y + blockDim.y*blockIdx.y;
	const int s = threadIdx.z + blockDim.z*blockIdx.z;
	if (c >= volSize.x || r >= volSize.y || s >= volSize.z) return;
	const int i = flatten(c, r, s, volSize.x, volSize.y, volSize.z);

	const int s_c = threadIdx.x + 1;
	const int s_r = threadIdx.y + 1;
	const int s_s = threadIdx.z + 1;
	const int s_w = blockDim.x + 2;
	const int s_h = blockDim.y + 2;
	const int s_t = blockDim.z + 2;
	const int s_i = flatten(s_c, s_r, s_s, s_w, s_h, s_t);

	extern __shared__ float s_float[];

	s_float[s_i] = buf1[i];

	if (threadIdx.x < 1)
	{
		s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] = buf1[flatten(c - 1, r, s, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c + blockDim.x, s_r, s_s, s_w, s_h, s_t)] = buf1[flatten(c + blockDim.x, r, s, volSize.x, volSize.y, volSize.z)];
	}
	if (threadIdx.y < 1)
	{
		s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] = buf1[flatten(c, r - 1, s, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c, s_r + blockDim.y, s_s, s_w, s_h, s_t)] = buf1[flatten(c, r + blockDim.y, s, volSize.x, volSize.y, volSize.z)];
	}
	if (threadIdx.z < 1)
	{
		s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)] = buf1[flatten(c, r, s - 1, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c, s_r, s_s + blockDim.z, s_w, s_h, s_t)] = buf1[flatten(c, r, s + blockDim.z, volSize.x, volSize.y, volSize.z)];
	}
	__syncthreads();

	if (s_float[s_i] < 0.f && (
		s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] > 0.f || s_float[flatten(s_c + 1, s_r, s_s, s_w, s_h, s_t)] > 0.f
		|| s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] > 0.f || s_float[flatten(s_c, s_r + 1, s_s, s_w, s_h, s_t)] > 0.f
		|| s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)] > 0.f || s_float[flatten(s_c, s_r, s_s + 1, s_w, s_h, s_t)] > 0.f))
	{
		buf2[i] = -1.f;
		buf2[flatten(c - 1, r, s, volSize.x, volSize.y, volSize.z)] = -1.f;
		buf2[flatten(c, r - 1, s, volSize.x, volSize.y, volSize.z)] = -1.f;
		buf2[flatten(c, r, s - 1, volSize.x, volSize.y, volSize.z)] = -1.f;
		buf2[flatten(c + 1, r, s, volSize.x, volSize.y, volSize.z)] = -1.f;
		buf2[flatten(c, r + 1, s, volSize.x, volSize.y, volSize.z)] = -1.f;
		buf2[flatten(c, r, s + 1, volSize.x, volSize.y, volSize.z)] = -1.f;
	}
}

__global__
void eroseKernel(float *buf2, float *buf1, int3 volSize)
{
	const int c = threadIdx.x + blockDim.x*blockIdx.x;
	const int r = threadIdx.y + blockDim.y*blockIdx.y;
	const int s = threadIdx.z + blockDim.z*blockIdx.z;
	if (c >= volSize.x || r >= volSize.y || s >= volSize.z) return;
	const int i = flatten(c, r, s, volSize.x, volSize.y, volSize.z);

	const int s_c = threadIdx.x + 1;
	const int s_r = threadIdx.y + 1;
	const int s_s = threadIdx.z + 1;
	const int s_w = blockDim.x + 2;
	const int s_h = blockDim.y + 2;
	const int s_t = blockDim.z + 2;
	const int s_i = flatten(s_c, s_r, s_s, s_w, s_h, s_t);

	extern __shared__ float s_float[];

	s_float[s_i] = buf1[i];

	if (threadIdx.x < 1)
	{
		s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] = buf1[flatten(c - 1, r, s, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c + blockDim.x, s_r, s_s, s_w, s_h, s_t)] = buf1[flatten(c + blockDim.x, r, s, volSize.x, volSize.y, volSize.z)];
	}
	if (threadIdx.y < 1)
	{
		s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] = buf1[flatten(c, r - 1, s, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c, s_r + blockDim.y, s_s, s_w, s_h, s_t)] = buf1[flatten(c, r + blockDim.y, s, volSize.x, volSize.y, volSize.z)];
	}
	if (threadIdx.z < 1)
	{
		s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)] = buf1[flatten(c, r, s - 1, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c, s_r, s_s + blockDim.z, s_w, s_h, s_t)] = buf1[flatten(c, r, s + blockDim.z, volSize.x, volSize.y, volSize.z)];
	}
	__syncthreads();

	if (s_float[s_i] < 0.f)
	{
		if (s == volSize.z - 1)
		{
			if (s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] > 0.f || s_float[flatten(s_c + 1, s_r, s_s, s_w, s_h, s_t)] > 0.f
				|| s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] > 0.f || s_float[flatten(s_c, s_r + 1, s_s, s_w, s_h, s_t)] > 0.f
				|| s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)] > 0.f)
			{
				buf2[i] = 1.f;
				buf2[flatten(c - 1, r, s, volSize.x, volSize.y, volSize.z)] = 1.f;
				buf2[flatten(c, r - 1, s, volSize.x, volSize.y, volSize.z)] = 1.f;
				buf2[flatten(c, r, s - 1, volSize.x, volSize.y, volSize.z)] = 1.f;
				buf2[flatten(c + 1, r, s, volSize.x, volSize.y, volSize.z)] = 1.f;
				buf2[flatten(c, r + 1, s, volSize.x, volSize.y, volSize.z)] = 1.f;
			}
			else buf2[i] = s_float[s_i];
		}
		else if (s == 0)
		{
			if (s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] > 0.f || s_float[flatten(s_c + 1, s_r, s_s, s_w, s_h, s_t)] > 0.f
				|| s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] > 0.f || s_float[flatten(s_c, s_r + 1, s_s, s_w, s_h, s_t)] > 0.f
				|| s_float[flatten(s_c, s_r, s_s + 1, s_w, s_h, s_t)] > 0.f)
			{
				buf2[i] = 1.f;
				buf2[flatten(c - 1, r, s, volSize.x, volSize.y, volSize.z)] = 1.f;
				buf2[flatten(c, r - 1, s, volSize.x, volSize.y, volSize.z)] = 1.f;
				buf2[flatten(c + 1, r, s, volSize.x, volSize.y, volSize.z)] = 1.f;
				buf2[flatten(c, r + 1, s, volSize.x, volSize.y, volSize.z)] = 1.f;
				buf2[flatten(c, r, s + 1, volSize.x, volSize.y, volSize.z)] = 1.f;
			}
			else buf2[i] = s_float[s_i];
		}
		else
		{
			if (s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] > 0.f || s_float[flatten(s_c + 1, s_r, s_s, s_w, s_h, s_t)] > 0.f
				|| s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] > 0.f || s_float[flatten(s_c, s_r + 1, s_s, s_w, s_h, s_t)] > 0.f
				|| s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)] > 0.f || s_float[flatten(s_c, s_r, s_s + 1, s_w, s_h, s_t)] > 0.f)
			{
				buf2[i] = 1.f;
				buf2[flatten(c - 1, r, s, volSize.x, volSize.y, volSize.z)] = 1.f;
				buf2[flatten(c, r - 1, s, volSize.x, volSize.y, volSize.z)] = 1.f;
				buf2[flatten(c, r, s - 1, volSize.x, volSize.y, volSize.z)] = 1.f;
				buf2[flatten(c + 1, r, s, volSize.x, volSize.y, volSize.z)] = 1.f;
				buf2[flatten(c, r + 1, s, volSize.x, volSize.y, volSize.z)] = 1.f;
				buf2[flatten(c, r, s + 1, volSize.x, volSize.y, volSize.z)] = 1.f;
			}
			else buf2[i] = s_float[s_i];
		}
	}
	else buf2[i] = s_float[s_i];
}

__global__
void boneCleanKernel(float *buf2, float *buf1, int3 volSize)
{
	const int c = threadIdx.x + blockDim.x*blockIdx.x;
	const int r = threadIdx.y + blockDim.y*blockIdx.y;
	const int s = threadIdx.z + blockDim.z*blockIdx.z;
	if (c >= volSize.x || r >= volSize.y || s >= volSize.z) return;
	const int i = flatten(c, r, s, volSize.x, volSize.y, volSize.z);

	const int s_c = threadIdx.x + 1;
	const int s_r = threadIdx.y + 1;
	const int s_s = threadIdx.z + 1;
	const int s_w = blockDim.x + 2;
	const int s_h = blockDim.y + 2;
	const int s_t = blockDim.z + 2;
	const int s_i = flatten(s_c, s_r, s_s, s_w, s_h, s_t);

	extern __shared__ float s_float[];

	s_float[s_i] = buf1[i];

	if (threadIdx.x < 1)
	{
		s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] = buf1[flatten(c - 1, r, s, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c + blockDim.x, s_r, s_s, s_w, s_h, s_t)] = buf1[flatten(c + blockDim.x, r, s, volSize.x, volSize.y, volSize.z)];
	}
	if (threadIdx.y < 1)
	{
		s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] = buf1[flatten(c, r - 1, s, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c, s_r + blockDim.y, s_s, s_w, s_h, s_t)] = buf1[flatten(c, r + blockDim.y, s, volSize.x, volSize.y, volSize.z)];
	}
	if (threadIdx.z < 1)
	{
		s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)] = buf1[flatten(c, r, s - 1, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c, s_r, s_s + blockDim.z, s_w, s_h, s_t)] = buf1[flatten(c, r, s + blockDim.z, volSize.x, volSize.y, volSize.z)];
	}
	__syncthreads();

	if (s_float[s_i] == 0.f)
	{
		if (s == volSize.z - 1)
		{
			if (s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] < 1.f && s_float[flatten(s_c + 1, s_r, s_s, s_w, s_h, s_t)] < 1.f
				&& s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] < 1.f && s_float[flatten(s_c, s_r + 1, s_s, s_w, s_h, s_t)] < 1.f
				&& s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)] < 1.f)
			{
				buf2[i] = -1.f;
			}
			else
			{
				buf2[i] = 1.f;
			}
		}
		else if (s == 0)
		{
			if (s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] < 1.f && s_float[flatten(s_c + 1, s_r, s_s, s_w, s_h, s_t)] < 1.f
				&& s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] < 1.f && s_float[flatten(s_c, s_r + 1, s_s, s_w, s_h, s_t)] < 1.f
				&& s_float[flatten(s_c, s_r, s_s + 1, s_w, s_h, s_t)] < 1.f)
			{
				buf2[i] = -1.f;
			}
			else
			{
				buf2[i] = 1.f;
			}
		}
		else
		{
			if (s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] < 1.f && s_float[flatten(s_c + 1, s_r, s_s, s_w, s_h, s_t)] < 1.f
				&& s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] < 1.f && s_float[flatten(s_c, s_r + 1, s_s, s_w, s_h, s_t)] < 1.f
				&& s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)] < 1.f && s_float[flatten(s_c, s_r, s_s + 1, s_w, s_h, s_t)] < 1.f)
			{
				buf2[i] = -1.f;
			}
			else
			{
				buf2[i] = 1.f;
			}
		}
	}
}

__global__
void findBondaryKernel(float *buf2, float *buf1, int3 volSize)
{
	const int c = threadIdx.x + blockDim.x*blockIdx.x;
	const int r = threadIdx.y + blockDim.y*blockIdx.y;
	const int s = threadIdx.z + blockDim.z*blockIdx.z;
	if (c >= volSize.x || r >= volSize.y || s >= volSize.z) return;
	const int i = flatten(c, r, s, volSize.x, volSize.y, volSize.z);

	const int s_c = threadIdx.x + 1;
	const int s_r = threadIdx.y + 1;
	const int s_s = threadIdx.z + 1;
	const int s_w = blockDim.x + 2;
	const int s_h = blockDim.y + 2;
	const int s_t = blockDim.z + 2;
	const int s_i = flatten(s_c, s_r, s_s, s_w, s_h, s_t);

	extern __shared__ float s_float[];

	s_float[s_i] = buf1[i];

	if (threadIdx.x < 1)
	{
		s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] = buf1[flatten(c - 1, r, s, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c + blockDim.x, s_r, s_s, s_w, s_h, s_t)] = buf1[flatten(c + blockDim.x, r, s, volSize.x, volSize.y, volSize.z)];
	}
	if (threadIdx.y < 1)
	{
		s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] = buf1[flatten(c, r - 1, s, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c, s_r + blockDim.y, s_s, s_w, s_h, s_t)] = buf1[flatten(c, r + blockDim.y, s, volSize.x, volSize.y, volSize.z)];
	}
	if (threadIdx.z < 1)
	{
		s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)] = buf1[flatten(c, r, s - 1, volSize.x, volSize.y, volSize.z)];
		s_float[flatten(s_c, s_r, s_s + blockDim.z, s_w, s_h, s_t)] = buf1[flatten(c, r, s + blockDim.z, volSize.x, volSize.y, volSize.z)];
	}
	__syncthreads();

	if (s_float[s_i] < 0.f)
	{
		if (s == volSize.z - 1)
		{
			if (s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] >= 1.f || s_float[flatten(s_c + 1, s_r, s_s, s_w, s_h, s_t)] >= 1.f
				|| s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] >= 1.f || s_float[flatten(s_c, s_r + 1, s_s, s_w, s_h, s_t)] >= 1.f
				|| s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)] >= 1.f)
			{
				buf2[i] = 0.f;
			}
			else buf2[i] = -1.f;
		}
		else if (s == 0)
		{
			if (s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] >= 1.f || s_float[flatten(s_c + 1, s_r, s_s, s_w, s_h, s_t)] >= 1.f
				|| s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] >= 1.f || s_float[flatten(s_c, s_r + 1, s_s, s_w, s_h, s_t)] >= 1.f
				|| s_float[flatten(s_c, s_r, s_s + 1, s_w, s_h, s_t)] >= 1.f)
			{
				buf2[i] = 0.f;
			}
			else buf2[i] = -1.f;
		}
		else
		{
			if (s_float[flatten(s_c - 1, s_r, s_s, s_w, s_h, s_t)] >= 1.f || s_float[flatten(s_c + 1, s_r, s_s, s_w, s_h, s_t)] >= 1.f
				|| s_float[flatten(s_c, s_r - 1, s_s, s_w, s_h, s_t)] >= 1.f || s_float[flatten(s_c, s_r + 1, s_s, s_w, s_h, s_t)] >= 1.f
				|| s_float[flatten(s_c, s_r, s_s - 1, s_w, s_h, s_t)] >= 1.f || s_float[flatten(s_c, s_r, s_s + 1, s_w, s_h, s_t)] >= 1.f)
			{
				buf2[i] = 0.f;
			}
			else buf2[i] = -1.f;
		}
	}
}