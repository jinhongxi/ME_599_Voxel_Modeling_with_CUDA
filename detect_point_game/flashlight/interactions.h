#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#define W 600
#define H 600
#define DELTA 5
#define TITLE_STRING "Flashlight: Find out the point"

int2 loc = { W / 2, H / 2 };
int2 loc0 = { W / 2, H / 2 };
bool dragMode = true;

void keyboard(unsigned char key, int x, int y)
{
	if (key == 'a') dragMode = !dragMode;
	if (key == 32) loc0.x = x; loc0.y = y;
	if (key == 13)
	{
		if ((abs(loc0.x - x) <= 5) && (abs(loc0.y - y) <= 5))
		{
			printf("Point FOUND! \n");
			exit(0);
		}
		else
		{
			printf("Please try again. \n");
		}
	}
	if (key == 27) exit(0);
	glutPostRedisplay();
}

void moseMove(int x, int y)
{
	if (dragMode) return;
	loc.x = x;
	loc.y = y;
	glutPostRedisplay();
}

void mouseDrag(int x, int y)
{
	if (!dragMode) return;
	loc.x = x;
	loc.y = y;
	glutPostRedisplay();
}

void handelSpecialKeypress(int key, int x, int y)
{
	if (key == GLUT_KEY_LEFT) loc.x -= DELTA;
	if (key == GLUT_KEY_RIGHT) loc.x += DELTA;
	if (key == GLUT_KEY_UP) loc.y -= DELTA;
	if (key == GLUT_KEY_DOWN) loc.y += DELTA;
	glutPostRedisplay();
}

void printInstructions()
{
	printf("Flastlight Interactions \n");
	printf("a: toggle mouse tracking mode \n");
	printf("space key: set the ref. point \n");
	printf("enter key: define the definition \n");
	printf("arrow keys: move center location \n");
	printf("esc: close graphics window \n");
}

#endif