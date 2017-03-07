#include "cuda_func.cuh"

#define STEP 0.05f

int divUp(int a, int A)
{
	return (a + A - 1) / A;
}

__device__
int flatten(int ch, int3 i, int4 volSize)
{
	return volSize.w*volSize.y*volSize.x*i.z + volSize.w*volSize.x*i.y + volSize.w*i.x + ch;
}

__device__
int clipWithBounds(int n, int n_min, int n_max)
{
	return n > n_max ? n_max : (n < n_min ? n_min : n);
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
int3 posToVolIndex(float3 pos, int4 volSize)
{
	return make_int3(pos.x + volSize.x / 2, pos.y + volSize.y / 2, pos.z + volSize.z / 2);
}

__device__
float3 paramRay(Ray r, float t)
{
	return r.o + t*(r.d);
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
uchar4 rayCastShader(Npp8u *d_bound, int4 volSize, Ray boxRay, float dist)
{
	uchar4 shade = { 0, 0, 0, 0 };
	float len = length(boxRay.d);
	float3 pos = boxRay.o;

	for (float t = 0.f; t < 1.f; t += STEP)
	{
		int3 index = posToVolIndex(pos, volSize);
		float d = sqrtf(1 - t*t) / 3;
		if (index.z == 0 || index.z == volSize.z - 1) d *= 3;
		shade.x += (int)d_bound[flatten(0, index, volSize)] * d;
		shade.y += (int)d_bound[flatten(1, index, volSize)] * d;
		shade.z += (int)d_bound[flatten(2, index, volSize)] * d;
		if (shade.x + shade.y + shade.z >= 100) break;
		pos = paramRay(boxRay, t);
	}

	return shade;
}

__global__
void renderFloatKernel(uchar4 *d_out, Npp8u *d_bound, int w, int h, int4 volSize, float alpha, float theta, float gamma, float dist)
{
	const int c = blockIdx.x*blockDim.x + threadIdx.x;
	const int r = blockIdx.y*blockDim.y + threadIdx.y;
	const int i = c + r * w;
	if ((c >= w) || (r >= h)) return;

	const uchar4 background = { 0, 0, 0, 0 };
	float3 source = { 0.f, 0.f, -dist };
	float3 pix = scrIdxToPos(c, r, w, h, 2 * volSize.z - dist);
	source = xRotate(source, alpha);
	source = yRotate(source, theta);
	source = zRotate(source, gamma);
	pix = xRotate(pix, alpha);
	pix = yRotate(pix, theta);
	pix = zRotate(pix, gamma);

	float t0, t1;
	const Ray pixRay = { source, pix - source };
	float3 center = { volSize.x / 2.f, volSize.y / 2.f, volSize.z / 2.f };
	const float3 boxmin = -center;
	const float3 boxmax = { volSize.x - center.x, volSize.y - center.y, volSize.z - center.z };
	const bool hitBox = intersectBox(pixRay, boxmin, boxmax, &t0, &t1);
	uchar4 shade;

	if (!hitBox) shade = background;
	else
	{
		if (t0 < 0.0f) t0 = 0.f;
		const Ray boxRay = { paramRay(pixRay, t0), paramRay(pixRay, t1) - paramRay(pixRay, t0) };
		shade = rayCastShader(d_bound, volSize, boxRay, dist);
	}

	d_out[i] = shade;
}