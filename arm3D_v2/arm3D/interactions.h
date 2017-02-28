#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#define IMG_W 114
#define IMG_H 134
#define IMG_T 54
#define W 800
#define H 800
#define TITLE_STRING "ME598C: Arm Segmentation"

float *b_vol = 0, *m_vol = 0, *f_vol = 0;
uchar4 *d_in = 0;
int3 volSize = { IMG_W, IMG_H, IMG_T };

#endif