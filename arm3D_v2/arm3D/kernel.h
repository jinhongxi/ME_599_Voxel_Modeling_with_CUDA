#ifndef KERNEL_H
#define KERNEL_H

struct uchar4;
struct int2;
struct unit4;

void importLauncher(uchar4 *d_in, int3 volSize);

void boneKernelLauncher(uchar4 *d_in, float *d_vol, int3 volSize);

void muscleKernelLauncher(uchar4 *d_in, float *d_vol, int3 volSize);

void fatKernelLauncher(uchar4 *d_in, float *d_vol, int3 volSize);

void exportLauncher(uchar4 *d_in, int3 volSize);

#endif