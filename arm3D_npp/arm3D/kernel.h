#ifndef KERNEL_H
#define KERNEL_H

#include <npp.h>
#include "cuda_func.cuh"
#include <stdio.h>
#include "CImg.h"

void importNPP(Npp8u *d_img, int4 imgSize, int4 volSize);

void exportNPP(Npp8u *d_img, int4 volSize);

void colorSeparateNPP(Npp8u *d_img, Npp8u *d_bone, Npp8u *d_muscle, Npp8u *d_fat, Npp8u *d_skin, int4 volSize);

void imageAddNPP(Npp8u *d_img, Npp8u *d_bone, Npp8u *d_muscle, Npp8u *d_fat, Npp8u *d_skin, int4 volSize);

void boneNPP(Npp8u *d_bone, int *boneDandE, int4 volSize);

void muscleNPP(Npp8u *d_muscle, int *muscleDandE, int4 volSize);

void fatNPP(Npp8u *d_fat, int blendDist, int4 volSize);

void skinNPP(Npp8u *d_skin, Npp8u *d_fat, int skinThickness, int4 volSize);

void trimNPP(Npp8u *d_bone, Npp8u *d_muscle, Npp8u *d_fat, Npp8u *d_skin, int4 volSize);

void nppLauncher(Npp8u *d_img, Npp8u *d_bone, Npp8u *d_muscle, Npp8u *d_fat, Npp8u *d_skin, int *boneDandE, int *muscleDandE, int blendDist, int skinThickness, int4 volSize);

void boundaryLauncher(Npp8u *d_bound, Npp8u *d_bone, Npp8u *d_muscle, Npp8u *d_fat, Npp8u *d_skin, int4 volSize, bool showBone, bool showMuscle, bool showFat, bool showSkin);

#endif