#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif
#include <vector_types.h>
#include "kernel.h"

#define W 800
#define H 800
#define TITLE_STRING "ME598C: Arm Segmentation"

#define IMG_W 114
#define IMG_H 134
#define IMG_T 54
#define IMG_CC 3
#define VOL_W 114
#define VOL_H 134
#define VOL_T 54
#define DELTA 5

const int4 imgSize = { IMG_W, IMG_H, IMG_T, IMG_CC };
const int4 volSize = { VOL_W, VOL_H, VOL_T, IMG_CC };
Npp8u *d_img = 0, *d_bone = 0, *d_muscle = 0, *d_fat = 0, *d_skin = 0, *d_bound = 0;
cudaEvent_t start, stop;

bool print = true, showBone = true, showMuscle = true, showFat = true, showSkin = true;
float dist = volSize.z * 1.5f, theta = 0.f, alpha = 1.f, gamma = -0.5f;
int boneDandE[8] = { 6, 0, 0, 3, 0, 0, 0, 0 };
int muscleDandE[8] = { 5, 0, 0, 3, 0, 5, 5, 0 };
int skinThickness = 1;
int blendDist = 5;

void mymenu(int value)
{
	switch (value)
	{
	case 0: return;
	case 1: showBone = !showBone; break;
	case 2: showMuscle = !showMuscle; break;
	case 3: showFat = !showFat; break;
	case 4: showSkin = !showSkin; break;
	}
	boundaryLauncher(d_bound, d_bone, d_muscle, d_fat, d_skin, volSize, showBone, showMuscle, showFat, showSkin);
	glutPostRedisplay();
}

void createMenu()
{
	glutCreateMenu(mymenu);
	glutAddMenuEntry("Show/Hide Tissues", 0);
	glutAddMenuEntry("Bone", 1);
	glutAddMenuEntry("Muscle", 2);
	glutAddMenuEntry("Fat", 3);
	glutAddMenuEntry("Skin", 4);
	glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void keyboard(unsigned char key, int x, int y)
{
	if (key == '-') dist += DELTA;
	if (key == '+') dist -= DELTA;
	if (key == 60) theta -= 0.1f;
	if (key == 62) theta += 0.1f;
	if (key == 8) dist = volSize.z * 1.5f, theta = 0.f, alpha = 1.f, gamma = -0.5f;
	if (key == 32) print = true;
	if (key == 27) exit(0);
	glutPostRedisplay();
}

void handleSpecialKeypress(int key, int x, int y)
{
	if (key == GLUT_KEY_LEFT) gamma -= 0.1f;
	if (key == GLUT_KEY_RIGHT) gamma += 0.1f;
	if (key == GLUT_KEY_UP) alpha += 0.1f;
	if (key == GLUT_KEY_DOWN) alpha -= 0.1f;
	glutPostRedisplay();
}

void printInstructions()
{
	printf("  Arg Segmentation Visualizer:\n\n"
		"    Zoom in/out             : + / -\n"
		"    Rotate in x-direction   : > / <\n"
		"    Rotate in y-direction   : Up / Down\n"
		"    Rotate in z-direction   : Right / Left\n"
		"    Reset parameters        : Backspace\n"
		"    Update figures          : Space\n"
		"    Exist                   : Esc\n"
		"    Right-click to show/hide tissues. \n");
}

#endif