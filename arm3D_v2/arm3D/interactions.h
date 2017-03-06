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
#define OUT_W 100
#define OUT_H 100
#define OUT_T 30

#define DELTA 5

cudaEvent_t start, stop;
float *b_vol = 0, *m_vol = 0, *f_vol = 0, *d_vol = 0;
uchar4 *d_in = 0;
const int3 volSize = { IMG_W, IMG_H, IMG_T + 2 };
const int3 parSize = { IMG_W * 2.f, IMG_H * 4.f, IMG_T * 8.f };
const int3 outSize = { OUT_W, OUT_H, OUT_T };
float zs = parSize.z / 2.f;
float dist = 0.f, theta = 0.f, alpha = 1.f, gamma = -0.5f;
bool showBone = true, showMuscle = true, showFat = true;

void mymenu(int value)
{
	switch (value)
	{
	case 0: return;
	case 1: showBone = !showBone; break;
	case 2: showMuscle = !showMuscle; break;
	case 3: showFat = !showFat; break;
	}
	volumeKernelLauncher(d_vol, b_vol, m_vol, f_vol, volSize, showBone, showMuscle, showFat);
	glutPostRedisplay();
}

void createMenu()
{
	glutCreateMenu(mymenu);
	glutAddMenuEntry("Show/Hide Tissues", 0);
	glutAddMenuEntry("Bone", 1);
	glutAddMenuEntry("Muscle", 2);
	glutAddMenuEntry("Fat", 3);
	glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void keyboard(unsigned char key, int x, int y)
{
	if (key == '+') zs -= DELTA;
	if (key == '-') zs += DELTA;
	if (key == 8) zs = parSize.z / 2.f, theta = 0.f, alpha = 0.f, dist = 0.f; // reset values
	if (key == 27) exit(0);
	if (key == 60) theta -= 0.1f;
	if (key == 62) theta += 0.1f;
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
		"    Exist                   : Esc\n"
		"    Right-click for object selection menu\n");
}

#endif