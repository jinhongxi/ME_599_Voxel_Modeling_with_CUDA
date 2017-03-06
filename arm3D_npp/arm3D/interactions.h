#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#define DISP_W 800
#define DISP_H 800
#define IMG_W 114
#define IMG_H 134
#define IMG_T 54
#define IMG_CC 3
#define VOL_W 114
#define VOL_H 134
#define VOL_T 54

const int4 imgSize = { IMG_W, IMG_H, IMG_T, IMG_CC };
const int4 volSize = { VOL_W, VOL_H, VOL_T, IMG_CC };
Npp8u *d_img = 0, *d_bone = 0, *d_muscle = 0, *d_fat = 0, *d_skin = 0;

int boneDandE[3] = { 2, 2, 7 };
int muscleDandE[3] = { 3, 2, 6 };
int skinThickness = 1;
int blendDist = 5;

#endif