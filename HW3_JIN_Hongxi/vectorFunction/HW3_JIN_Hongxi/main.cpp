/*
Homework 3 , ME 599 H , WI 2017
Name : JIN Hongxi
Student ID : 1628292
UW NetID : jinhx
*/

#include "hw3.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define MIN 32
#define MAX 50000000

int main()
{
	FILE *out = fopen("hw3.csv", "w");
	fprintf(out, "Array Length, Dot Product, Time Used [Ms], Atomic\n");

	for (int atomic = 1; atomic > -2; --atomic)
	{
		for (int N = MIN; N <= MAX; N += N / 2)
		{
			float *u2 = (float*)calloc(N, sizeof(float));
			float *v2 = (float*)calloc(N, sizeof(float));
			float dotP = 0.0f;
			float time = 0.0f;

			valueAssignment(u2, N, 0.25f);
			valueAssignment(v2, N, 0.75f);

			vec_fun(&time, &dotP, u2, v2, N, 1.0f, 0.0f, 0.0f, 0.0f, atomic);

			fprintf(out, "%i, %f, %f, %i\n", N, dotP, time, atomic);

			free(u2);
			free(v2);
		}
	}

	fclose(out); 

	return 0;
}