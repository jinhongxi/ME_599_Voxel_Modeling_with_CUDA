#ifndef C_FUNC_H
#define C_FUNC_H

#define IMG_W 512
#define IMG_H 512
#define IMG_T 138
#define TITLE_STRING "Nasal Analysis"

const int3 imgSize = { IMG_W, IMG_H, IMG_T };
Npp16u *d_in = 0, *d_out = 0;

#endif