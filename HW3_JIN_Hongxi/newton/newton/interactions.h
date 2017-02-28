#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#define W 600
#define H 600
#define DELTA 1
#define TITLE_STRING "Newton's Method"

int iteration = 2;
int syst = 2;
int2 loc = { W / 2, H / 2 };
bool dragMode = true;

void keyboard(unsigned char key, int x, int y)
{
	if (key == 13) dragMode = !dragMode;
	if (key == '1') { syst = 1; printf("System = %i:\n", syst); }
	if (key == '2') { syst = 2; printf("System = %i:\n", syst); }
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
	if (key == GLUT_KEY_DOWN)
	{
		iteration -= DELTA;
		if (iteration <= 1) iteration = 1;
		printf("Iteration=%i:\n", iteration);
	}
	if (key == GLUT_KEY_UP)
	{
		iteration += DELTA;
		printf("Iteration=%i:\n", iteration);
	}
	glutPostRedisplay();
}

void printInstructions()
{
	printf("Newton's Method: \n\n");
	printf("up/down arrow keys: change iterations \n");
	printf("number keys: change system model\n");
	printf("  1: f(x)=x^3-x\n");
	printf("  2: f(z)=z^3-1\n");
	printf("Enter: fix/release mouse position \n");
	printf("Esc: close graphics window \n\n");
}

#endif