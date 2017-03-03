#ifndef KERNEL_H
#define KERNEL_H

struct uchar4;
struct int2;
struct unit4;

void importLauncher(uchar4 *d_in, int3 volSize, int3 outSize);

void boneKernelLauncher(uchar4 *d_in, float *d_vol, int3 volSize);

void muscleKernelLauncher(uchar4 *d_in, float *d_vol, int3 volSize);

void fatKernelLauncher(uchar4 *d_in, float *d_vol, float *m_vol, float *b_vol, int3 volSize);

void exportLauncher(uchar4 *d_in, int3 volSize);

void volumeKernelLauncher(float *d_vol, float *b_vol, float *m_vol, float *f_vol, int3 volSize, bool b_disp, bool m_disp, bool f_disp);

void kernelLauncher(uchar4 *d_out, uchar4 *d_in, float *d_vol, int w, int h, int3 volSize, int3 parSize, float zs, float alpha, float theta, float gamma, float dist);

#endif