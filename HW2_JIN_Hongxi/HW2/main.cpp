/*
Homework 2 , ME 599 H , WI 2017
Name : JIN Hongxi
Student ID : 1628292
UW NetID : jinhx
*/

#include "hw2.h"
#include <stdio.h>
#include <stdlib.h>
#define N 32

int main()
{
	printf("\n Homework 2, ME 599 H, WI 2017 \n Name: JIN Hongxi \n Student ID: 1628292 \n UW NetID: jinhx \n");
	printf("\n INPUTS: \n");

	printf("   N = %i; \n\n", N);
	
	float *u2 = (float*)calloc(N, sizeof(float));
	float *v2 = (float*)calloc(N, sizeof(float));
	float *w2 = (float*)calloc(N, sizeof(float));

	printf("   u2 = { ");
	valueAssignment(u2, N, 0.25f);
	printf("} \n\n");
	printf("   v2 = { ");
	valueAssignment(v2, N, 0.75f);
	printf("} \n\n");

	printf("\n VECTOR OPERATIONS: \n\n");

	printf("   u2 + v2 = { ");
	vec_fun(w2, u2, v2, N, 0.0f, 1.0f, 1.0f, 0.0f);
	printf("} \n\n");

	printf("   -3u2 + v2 = { ");
	vec_fun(w2, u2, v2, N, 0.0f, -3.0f, 1.0f, 0.0f);
	printf("} \n\n");

	printf("   u2 * v2 = { ");
	vec_fun(w2, u2, v2, N, 1.0f, 0.0f, 0.0f, 0.0f);
	printf("} \n\n");

	printf("\n SCALE OPERATIONS: \n\n");
	printf("   u2 * v2 = { ");
	float product = scaleComputing(w2, u2, v2, N, 1);
	printf("} \n\n");
	printf("\n   Inner product of u2 and v2 = %.2f \n", product);

	product = scaleComputing(w2, u2, v2, N, 2);
	printf("   Revised inner product of u2 and v2 = %.2f \n\n", product);

	printf("   u2 + v2 = { ");
	product = scaleComputing(w2, u2, v2, N, 3);
	printf("} \n\n");
	printf("\n   Euclidean norm of u2+v2 = %.2f \n", product);

	printf("\n Press any key to continue. \n");
	getchar();

	free(u2);
	free(v2);
	free(w2);

	return 0;
}