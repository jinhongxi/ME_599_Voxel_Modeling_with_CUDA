#ifndef KERNEL_H
#define KERNEL_H

struct uchar4;
struct int2;
struct float2;

void kernelLauncher(float *x, uchar4 *d_out, int w, int h, int2 pos, int step, int sys);

#endif