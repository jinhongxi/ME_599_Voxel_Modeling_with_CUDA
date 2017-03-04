#ifndef KERNEL_H
#define KERNEL_H

#include <npp.h>
#include "cuda_func.cuh"
#include <stdio.h>
#include "CImg.h"

void importKernel(Npp8u *d_img, int4 imgSize, int4 volSize);

void exportKernel(Npp8u *d_img, int4 volSize);

#endif