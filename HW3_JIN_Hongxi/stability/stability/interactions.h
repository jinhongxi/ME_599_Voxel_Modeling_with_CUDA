#ifndef INTERACTIONS_H
#define INTERACTIONS_H
#define DELTA_P 0.1f
#define DELTA_T 0.005f
#define TITLE_STRING "Stability"
#define W 600
#define H 600

int sys = 0;
int ord = 4;
float param = 0.1f;
float step = 0.005f;
void keyboard(unsigned char key, int x, int y) {
	if (key == 27) exit(0);
	if (key == '0') sys = 0;
	if (key == '1') sys = 1;
	if (key == '2') sys = 2;
	if (key == 'e') ord = 1;
	if (key == 'r') ord = 4;
	glutPostRedisplay();
}

void handleSpecialKeypress(int key, int x, int y) {
	if (key == GLUT_KEY_DOWN) param -= DELTA_P;
	if (key == GLUT_KEY_UP) param += DELTA_P;
	if (key == GLUT_KEY_LEFT)
	{
		step -= DELTA_T;
		if (step < 0.005f) step = 0.005f;
	}
	if (key == GLUT_KEY_RIGHT)
	{
		step += DELTA_T;
		if (step > 1.0f) step = 1.0f;
	}
	glutPostRedisplay();
}

// no mouse interactions implemented for this app
void mouseMove(int x, int y) { return; }
void mouseDrag(int x, int y) { return; }

void printInstructions() {
	printf("Stability visualizer\n");
	printf("Use number keys to select system:\n");
	printf("\t0: linear oscillator: positive stiffness\n");
	printf("\t1: linear oscillator: negative stiffness\n");
	printf("\t2: van der Pol oscillator: nonlinear damping\n");
	printf("Use e/r keys to select numerical analysis method:\n");
	printf("\te: Euler's Method: First order\n");
	printf("\tr: Runge-Kutta's Method: Forth order\n");
	printf("Use up/down arrow keys to adjust parameter value\n\n");
	printf("Use right/left arrow keys to adjust time step\n\n");
	printf("Output:\n\n");
	printf("Paremeter, System, RK Order, Distance, Time Step, Steps Taken, Kernel Time\n");
}

#endif