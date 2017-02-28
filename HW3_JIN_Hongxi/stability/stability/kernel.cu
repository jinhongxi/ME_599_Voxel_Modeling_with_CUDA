#include "kernel.h"
#include <stdio.h>
#define TX 32
#define TY 32
#define LEN 5.f
#define FINAL_TIME 10.f

// scale coordinates onto [-LEN, LEN]
__device__
float scale(int i, int w) { return 2 * LEN*(((1.f*i) / w) - 0.5f); }

// function for right-hand side of y-equation
__device__
float f(float x, float y, float param, float sys) {
	if (sys == 1) return x - 2 * param*y; // negative stiffness
	if (sys == 2) return -x + param*(1 - x*x)*y; //van der Pol
	else return -x - 2 * param*y;
}

__device__
float2 diff(float x, float y, float dt, float param, float sys, int order)
{
	float dx = 0.f, dy = 0.f, kx = 0.f, ky = 0.f;

	switch (order)
	{
	case 4:
	{
		kx = y;
		ky = f(x, y, param, sys);
		dx += kx*dt / 6;
		dy += ky*dt / 6;

		kx = y + ky * dt / 2;
		ky = f(x + kx * dt / 2, y + ky * dt / 2, param, sys);
		dx += kx*dt / 3;
		dy += ky*dt / 3;

		kx = y + ky * dt / 2;
		ky = f(x + kx * dt / 2, y + ky * dt / 2, param, sys);
		dx += kx*dt / 3;
		dy += ky*dt / 3;

		kx = y + ky * dt;
		ky = f(x + kx * dt, y + ky * dt, param, sys);
		dx += kx*dt / 6;
		dy += ky*dt / 6;

		break;
	}
	default: case 1:
	{
		dx = y*dt;
		dy = f(x, y, param, sys)*dt;
		break;
	}
	}

	return make_float2(dx, dy);
}

// explicit Runge-Kutta solver
__device__
float2 rungekutta(float x, float y, float dt, float tFinal, float param, float sys, int order) 
{
	for (float t = 0; t < tFinal; t += dt) {
		float2 d = diff(x, y, dt, param, sys, order);
		x += d.x;
		y += d.y;
	}
	return make_float2(x, y);
}

__device__
unsigned char clip(float x)
{ 
	return x > 255 ? 255 : (x < 0 ? 0 : x); 
}

// kernel function to compute decay and shading
__global__
void stabImageKernel(uchar4 *d_out, int w, int h, float p, int s, int o, float dt) 
{
	const int c = blockIdx.x*blockDim.x + threadIdx.x;
	const int r = blockIdx.y*blockDim.y + threadIdx.y;
	if ((c >= w) || (r >= h)) return; // Check if within image bounds
	const int i = c + r*w; // 1D indexing
	const float x0 = scale(c, w);
	const float y0 = scale(r, h);
	const float dist_0 = sqrt(x0*x0 + y0*y0);
	const float2 pos = rungekutta(x0, y0, dt, FINAL_TIME, p, s, o);
	const float dist_f = sqrt(pos.x*pos.x + pos.y*pos.y);
	if ((c == w * 55 / 100) && (r == h * 55 / 100))
	{
		printf("%f,%i,%i,%f,%f,%i,", p, s, o, dist_f, dt, (int)(FINAL_TIME / dt));
	}
	// assign colors based on distance from origin
	const float dist_r = dist_f / dist_0;
	d_out[i].x = clip(dist_r * 255); // red ~ growth
	d_out[i].y = ((c == w / 2) || (r == h / 2)) ? 255 : 0; // axes
	d_out[i].z = clip((1 / dist_r) * 255); // blue ~ 1/growth
	d_out[i].w = 255;
}

void kernelLauncher(uchar4 *d_out, int w, int h, float p, int s, int o, float tStep) 
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	const dim3 blockSize(TX, TY);
	const dim3 gridSize = dim3((w + TX - 1) / TX, (h + TY - 1) / TY);
	stabImageKernel << <gridSize, blockSize >> >(d_out, w, h, p, s, o, tStep);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time = 0.0f;
	cudaEventElapsedTime(&time, start, stop);
	printf("%f \n", time);
}