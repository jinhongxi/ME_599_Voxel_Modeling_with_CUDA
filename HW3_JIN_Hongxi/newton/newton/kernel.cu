#include "kernel.h"
#include <stdio.h>
#include <math.h>
#define TX 32
#define TY 32
#define LENX 2.f
#define LENY 2.f
#define RADIUS 50
#define MAXITERATION 20

int kernelSize(int a, int A)
{
	return((a + A - 1) / A);
}

__device__
unsigned char clip(int n)
{
	return n > 255 ? 255 : (n < 0 ? 0 : n);
}

__device__
float2 intToScale (int a, int b, int A, int B)
{ 
	float sx = 2 * LENX*(((1.f*a) / A) - 0.5f);
	float sy = 2 * LENY*(0.5f - ((1.f*b) / B));
	return make_float2(sx, sy);
}

__device__
int2 scaleToInt(float2 z, int A, int B)
{
	float par = (z.x + LENX) / (2 * LENX);
	int ix = par*A;
	par = (LENY - z.y) / (2 * LENY);
	int iy = par*B;
	return make_int2(ix, iy);
}

__device__
float2 f(float2 z, int sys) 
{
	float fx = 0.f, fy = 0.f;
	if (sys == 1) { fx = z.x*z.x*z.x - z.x; }
	if (sys == 2)
	{
		fx = z.x*z.x*z.x - 3 * z.x*z.y*z.y - 1;
		fy = 3 * z.x*z.x*z.y - z.y*z.y*z.y;
	}
	return make_float2(fx, fy);
}

__device__
float2 df(float2 z, int sys)
{
	float dx = 0.f, dy = 0.f;
	if (sys == 1) { dx = 3 * z.x*z.x - 1; }
	if (sys == 2)
	{
		float fx = z.x*z.x*z.x - 3 * z.x*z.y*z.y - 1;
		float fy = 3 * z.x*z.x*z.y - z.y*z.y*z.y;
		float a = z.x*z.x - z.y*z.y;
		float b = 2 * z.x*z.y;
		float c = 3 * (z.x*z.x + z.y*z.y)*(z.x*z.x + z.y*z.y);
		dx = fx*c / (fx*a + fy*b);
		dy = fy*c / (fy*a - fx*b);
	}
	return make_float2(dx, dy);
}

__device__
float2 newton(float2 z, int step, int sys)
{
	float2 zk = make_float2(z.x, z.y);
	for (int n = 0; n < step; ++n)
	{
		float2 fk = f(zk, sys);
		float2 dfk = df(zk, sys);
		zk.x -= fk.x / dfk.x;
		zk.y -= fk.y / dfk.y;
	}
	return zk;
}

__device__
int newtonIteration(float2 z, int sys)
{
	int k = 0;
	float2 zk = make_float2(z.x, z.y);
	while (k <= MAXITERATION)
	{
		float2 fk = f(zk, sys);
		float2 dfk = df(zk, sys);
		zk.x -= fk.x / dfk.x;
		zk.y -= fk.y / dfk.y;
		if ((abs(fk.x) <= 0.1f) && (abs(fk.y) <= 0.1f)) break;
		k++;
	}
	return k;
}

__device__
void colorize(uchar4 *color, int i, int distin, int distout, int distori, int prop)
{
	switch (prop)
	{
	case 1:
	{
		const unsigned char c_in = clip(distin * RADIUS);
		const unsigned char c_out = clip(distout * RADIUS/2);
		const unsigned char c_ori = clip(distori * 255 / MAXITERATION);
		color[i].x = c_in + (255 - c_ori);
		color[i].y = c_ori + (255 - c_out);
		color[i].z = c_out;
		color[i].w = 255;
		break;
	}
	case 0:
	{
		color[i].x = 0, color[i].y = 0, color[i].z = 0, color[i].w = 255;
		break;
	}
	case -1:
	{
		color[i].x = 255, color[i].y = 0, color[i].z = 255, color[i].w = 255;
		break;
	}
	case -2:
	{
		color[i].x = 0, color[i].y = 0, color[i].z = 255, color[i].w = 255;
		break;
	}
	default:
	{
		color[i].x = 255, color[i].y = 255, color[i].z = 255, color[i].w = 255;
		break;
	}
	}
}

__global__
void newtonKernel(float *d_x, uchar4 *d_out, int w, int h, int2 pos, int step, int sys)
{
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
	if ((c >= w) || (r >= h)) return;
	const int i = c + r * w;

	const float2 f_z = intToScale(c, r, w, h);
	const float2 f_in = intToScale(pos.x, pos.y, w, h);
	const float2 f_out = newton(f_in, step, sys);
	const float2 f_curve = f(f_z, sys);
	const float2 f_diff = df(f_z, sys);

	if (sys == 1) { d_x[0] = f_in.x; d_x[1] = f_out.x; d_x[2] = abs(f_in.x - f_out.x); }
	if (sys == 2)
	{
		d_x[0] = f_in.x; d_x[1] = f_in.y;
		d_x[2] = f_out.x; d_x[3] = f_out.y;
		d_x[4] = sqrtf((f_out.x - f_in.x)*(f_out.x - f_in.x) + (f_out.y - f_in.y)*(f_out.y - f_in.y));
	}

	const int2 i_out = scaleToInt(f_out, w, h);
	const int2 i_curve = scaleToInt(f_curve, w, h);
	const int2 i_diff = scaleToInt(f_diff, w, h);
	
	int distin = sqrtf((c - pos.x)*(c - pos.x) + (r - pos.y)*(r - pos.y));
	int distout = sqrtf((c - i_out.x)*(c - i_out.x))*2;
	//int distori = sqrtf((i_curve.x - h / 2)*(i_curve.x - h / 2) + (r - h / 2)*(r - h / 2)) * 3;
	int distori = newtonIteration(f_z, sys);
	if (sys == 2)
	{
		distout = sqrtf((c - i_out.x)*(c - i_out.x) + (r - i_out.y)*(r - i_out.y));
		//distori = sqrtf((i_curve.x - w / 2)*(i_curve.x - w / 2) + (i_curve.y - h / 2)*(i_curve.y - h / 2));
	}

	if ((sys == 1) && (r == h - i_curve.x)) colorize(d_out, i, 0, 0, 0, 0);
	else if ((sys == 1) && (r == h - i_diff.x)) colorize(d_out, i, 0, 0, 0, 0);
	else if ((c == w / 2) || (r == h / 2)) colorize(d_out, i, 0, 0, 0, 0);
	else colorize(d_out, i, distin, distout, distori, 1);
}

void kernelLauncher(float *x, uchar4 *d_out, int w, int h, int2 pos, int step, int sys)
{
	float *d_x = 0;
	cudaMalloc(&d_x, 5 * sizeof(float));

	const dim3 blockSize(TX, TY);
	const dim3 gridSize = dim3(kernelSize(w, TX), kernelSize(h, TY));

	newtonKernel <<<gridSize, blockSize>>>(d_x, d_out, w, h, pos, step, sys);
	
	cudaMemcpy(x, d_x, 5 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_x);
}