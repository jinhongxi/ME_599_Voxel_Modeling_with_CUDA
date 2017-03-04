#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#define DISP_W 800
#define DISP_H 800
#define IMG_W 114
#define IMG_H 134
#define IMG_T 54
#define IMG_CC 3
#define VOL_W 200
#define VOL_H 200
#define VOL_T 50

const int4 imgSize = { IMG_W, IMG_H, IMG_T, IMG_CC };
const int4 volSize = { VOL_W, VOL_H, VOL_T, IMG_CC };

Npp8u *d_img = 0;

#endif