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
int clipWithBounds(int n, int n_min, int n_max) 
{
	return n > n_max ? n_max : (n < n_min ? n_min : n);
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

__device__
float3 xRotate(float3 pos, float theta)
{
	const float c = cosf(theta), s = sinf(theta);
	return make_float3(pos.x, c*pos.y - s*pos.z, s*pos.y + c*pos.z);
}

__device__ 
float3 yRotate(float3 pos, float theta) 
{
	const float c = cosf(theta), s = sinf(theta);
	return make_float3(s*pos.z + c*pos.x, pos.y, c*pos.z - s*pos.x);
}

__device__
float3 zRotate(float3 pos, float theta)
{
	const float c = cosf(theta), s = sinf(theta);
	return make_float3(c*pos.x - s*pos.y, s*pos.x + c*pos.y, pos.z);
}

__device__ 
float3 scrIdxToPos(int c, int r, int w, int h, float zs) 
{
	return make_float3(c - w / 2, r - h / 2, zs);
}

__device__ 
float3 paramRay(Ray r, float t) 
{ 
	return r.o + t*(r.d); 
}

__device__ 
float planeSDF(float3 pos, float3 norm, float d) 
{
	return dot(pos, normalize(norm)) - d;
}

__device__
bool rayPlaneIntersect(Ray myRay, float3 n, float dist, float *t) 
{
	const float f0 = planeSDF(paramRay(myRay, 0.f), n, dist);
	const float f1 = planeSDF(paramRay(myRay, 1.f), n, dist);
	bool result = (f0*f1 < 0);
	if (result) *t = (0.f - f0) / (f1 - f0);
	return result;
}

__device__ 
bool intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar) 
{
	const float3 invR = make_float3(1.0f) / r.d;
	const float3 tbot = invR*(boxmin - r.o), ttop = invR*(boxmax - r.o);
	const float3 tmin = fminf(ttop, tbot), tmax = fmaxf(ttop, tbot);
	*tnear = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
	*tfar = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));
	return *tfar > *tnear;
}

__device__ 
int3 posToVolIndex(float3 pos, int3 volSize) 
{
	return make_int3(pos.x + volSize.x / 2, pos.y + volSize.y / 2, pos.z + volSize.z / 2);
}

__device__
int posToImgIdx(float3 pos, int3 volSize, int3 parSize)
{
	int3 ind = posToVolIndex(pos, parSize);
	int3 delta = make_int3((parSize.x - volSize.x) / 2, (parSize.y - volSize.y) / 2, (parSize.z - volSize.z) / 2);
	return flatten(ind.x - delta.x, ind.y - delta.y, ind.z - delta.z, volSize.x, volSize.y, volSize.z);
}

__device__ 
float density(float *d_vol, int3 volSize, int3 parSize, float3 pos)
{
	int3 index = posToVolIndex(pos, volSize);
	int i = index.x, j = index.y, k = index.z;
	const int w = volSize.x, h = volSize.y, d = volSize.z;
	const float3 rem = fracf(pos);
	index = make_int3(clipWithBounds(i, 0, w - 2), clipWithBounds(j, 0, h - 2), clipWithBounds(k, 0, d - 2));

	const float dens000 = d_vol[flatten(index.x, index.y, index.z, volSize.x, volSize.y, volSize.z)];
	const float dens100 = d_vol[flatten(index.x + 1, index.y, index.z, volSize.x, volSize.y, volSize.z)];
	const float dens010 = d_vol[flatten(index.x, index.y + 1, index.z, volSize.x, volSize.y, volSize.z)];
	const float dens001 = d_vol[flatten(index.x, index.y, index.z + 1, volSize.x, volSize.y, volSize.z)];
	const float dens110 = d_vol[flatten(index.x + 1, index.y + 1, index.z, volSize.x, volSize.y, volSize.z)];
	const float dens101 = d_vol[flatten(index.x + 1, index.y, index.z + 1, volSize.x, volSize.y, volSize.z)];
	const float dens011 = d_vol[flatten(index.x, index.y + 1, index.z + 1, volSize.x, volSize.y, volSize.z)];
	const float dens111 = d_vol[flatten(index.x + 1, index.y + 1, index.z + 1, volSize.x, volSize.y, volSize.z)];
	return (1 - rem.x)*(1 - rem.y)*(1 - rem.z)*dens000 + (rem.x)*(1 - rem.y)*(1 - rem.z)*dens100 +
		(1 - rem.x)*(rem.y)*(1 - rem.z)*dens010 + (1 - rem.x)*(1 - rem.y)*(rem.z)*dens001 
		+ (rem.x)*(rem.y)*(1 - rem.z)*dens110 + (rem.x)*(1 - rem.y)*(rem.z)*dens101 
		+ (1 - rem.x)*(rem.y)*(rem.z)*dens011 + (rem.x)*(rem.y)*(rem.z)*dens111;
}

__device__
float func(int c, int r, int s, int3 volSize, float4 params)
{
	const int3 pos0 = { volSize.x / 2, volSize.y / 2, volSize.z / 2 };
	const float dx = c - pos0.x, dy = r - pos0.y, dz = s - pos0.z;

	float x = fabsf(dx) - params.x, y = fabsf(dy) - params.y, z = fabsf(dz) - params.z;
	if (x <= 0 && y <= 0 && z <= 0) return fmaxf(x, fmaxf(y, z));
	else
	{
		x = fmaxf(x, 0), y = fmaxf(y, 0), z = fmaxf(z, 0);
		return sqrtf(x*x + y*y + z*z);
	}
}

__device__ 
uchar4 rayCastShader(uchar4 *d_in, float *d_vol, float *b_vol, float *m_vol, float *f_vol, int3 volSize, int3 parSize, Ray boxRay, bool b_disp, bool m_disp, bool f_disp, float dist)
{
	uchar4 shade = make_uchar4(0, 0, 0, 0);
	float3 pos = boxRay.o;
	float len = length(boxRay.d);
	float t = 0.0f;
	float f = density(d_vol, volSize, parSize, pos);
	while (f > dist + EPS && t < 1.0f) 
	{
		f = density(d_vol, volSize, parSize, pos);
		t += (f - dist) / len;
		pos = paramRay(boxRay, t);
		f = density(d_vol, volSize, parSize, pos);
	}
	if (t < 1.f) 
	{
		const float3 ux = make_float3(1, 0, 0), uy = make_float3(0, 1, 0), uz = make_float3(0, 0, 1);
		float3 grad = { (density(d_vol, volSize, parSize, pos + EPS*ux) - density(d_vol, volSize, parSize, pos)) / EPS,
			(density(d_vol, volSize, parSize, pos + EPS*uy) - density(d_vol, volSize, parSize, pos)) / EPS,
			(density(d_vol, volSize, parSize, pos + EPS*uz) - density(d_vol, volSize, parSize, pos)) / EPS };
		float intensity = -dot(normalize(boxRay.d), normalize(grad));
		int  i = posToImgIdx(pos, volSize, parSize);
		shade = make_uchar4(d_in[i].x * intensity, d_in[i].y * intensity, d_in[i].z * intensity, 255 * intensity);
	}
	return shade;
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

__global__
void volumeKernel(float *d_vol, int3 volSize, float4 params)
{
	const int c = threadIdx.x + blockDim.x*blockIdx.x;
	const int r = threadIdx.y + blockDim.y*blockIdx.y;
	const int s = threadIdx.z + blockDim.z*blockIdx.z;
	if (c >= volSize.x || r >= volSize.y || s >= volSize.z) return;
	const int i = flatten(c, r, s, volSize.x, volSize.y, volSize.z);

	d_vol[i] = func(c, r, s, volSize, params);
}

__global__
void renderFloatKernel(uchar4 *d_out, uchar4 *d_in, float *d_vol, float *b_vol, float *m_vol, float *f_vol, int w, int h, int3 volSize, int3 parSize, float zs, float gamma, float theta, float alpha, bool b_disp, bool m_disp, bool f_disp, float dist)
{
	const int c = blockIdx.x*blockDim.x + threadIdx.x;
	const int r = blockIdx.y*blockDim.y + threadIdx.y;
	const int i = c + r * w;
	if ((c >= w) || (r >= h)) return;

	const uchar4 background = { 0, 0, 0, 0 };
	float3 source = { 0.f, 0.f, -zs };
	float3 pix = scrIdxToPos(c, r, w, h, 2 * parSize.z - zs);
	source = xRotate(source, alpha);
	source = yRotate(source, theta);
	source = zRotate(source, gamma);
	pix = xRotate(pix, alpha);
	pix = yRotate(pix, theta);
	pix = zRotate(pix, gamma);

	float t0, t1;
	const Ray pixRay = { source, pix - source };
	float3 center = { parSize.x / 2.f, parSize.y / 2.f, parSize.z / 2.f };
	const float3 boxmin = -center;
	const float3 boxmax = { parSize.x - center.x, parSize.y - center.y, parSize.z - center.z };
	const bool hitBox = intersectBox(pixRay, boxmin, boxmax, &t0, &t1);
	uchar4 shade;

	if (!hitBox) shade = background;
	else 
	{
		if (t0 < 0.0f) t0 = 0.f; 
		const Ray boxRay = { paramRay(pixRay, t0), paramRay(pixRay, t1) - paramRay(pixRay, t0) };
		shade = rayCastShader(d_in, d_vol, b_vol, m_vol, f_vol, volSize, parSize, boxRay, b_disp, m_disp, f_disp, dist);
	}

	d_out[i] = shade;
}