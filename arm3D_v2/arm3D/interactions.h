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

#define IMG_W 114
#define IMG_H 134
#define IMG_T 54
#define W 800
#define H 800
#define TITLE_STRING "ME598C: Arm Segmentation"

#define DELTA 5

float *b_vol = 0, *m_vol = 0, *f_vol = 0, *d_vol = 0;
uchar4 *d_in = 0;
const int3 volSize = { IMG_W, IMG_H, IMG_T };
const int3 parSize = { IMG_W * 4.f, IMG_H * 8.f, IMG_T * 16.f };
const float4 params = { IMG_W, IMG_H, IMG_T, 1.f };
float zs = parSize.z / 2.f;
float dist = 0.f, theta = 0.f, alpha = 1.f, gamma = -0.5f;
bool showBone = true, showMuscle = true, showFat = true, print = false;

void mymenu(int value)
{
	switch (value)
	{
	case 0: return;
	case 1: showBone = !showBone; break;
	case 2: showMuscle = !showMuscle; break;
	case 3: showFat = !showFat; break;
	}
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
	if (key == 8) zs = IMG_T, theta = 0.f, alpha = 0.f, dist = 0.f; // reset values
	if (key == 'b') showBone = !showBone;
	if (key == 'f') showMuscle = !showMuscle;
	if (key == 'f') showFat = !showFat;
	if (key == 32) print = true;
	if (key == 27) exit(0);
	if (key == 60) gamma -= 0.1f;
	if (key == 62) gamma += 0.1f;
	glutPostRedisplay();
}

void handleSpecialKeypress(int key, int x, int y)
{
	if (key == GLUT_KEY_LEFT) theta -= 0.1f;
	if (key == GLUT_KEY_RIGHT) theta += 0.1f;
	if (key == GLUT_KEY_UP) alpha += 0.1f;
	if (key == GLUT_KEY_DOWN) alpha -= 0.1f;
	glutPostRedisplay();
}

void printInstructions()
{
	printf("  Arg Segmentation Visualizer:\n\n"
		"    Show/Hide bone          : b\n"
		"    Show/Hide muscle        : m\n"
		"    Show/Hide fat           : f\n"
		"    Zoom in/out             : + / -\n"
		"    Rotate in x-direction   : Up / Down\n"
		"    Rotate in y-direction   : Right / Left\n"
		"    Rotate in z-direction   : > / <\n"
		"    Reset parameters        : Backspace\n"
		"    Print images            : Space\n"
		"    Exist                   : Esc\n"
		"    Right-click for object selection menu\n");
}

#endif