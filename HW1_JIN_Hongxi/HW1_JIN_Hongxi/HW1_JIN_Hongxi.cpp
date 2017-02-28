/*
Homework 1 , ME 599 H , WI 2017
Name : JIN Hongxi
Student ID : 1628292
UW NetID : jinhx
*/

#include <stdio.h>
#include <math.h>

/*---------------------------------------------------------------------------------------------------------*\
|  PART 1: "vec_fun" notes                                                                                  |
|                                                                                                           |
|  INPUTS:                                                                                                  |
|  - x1, x2 = floating-point arrays with the same length;                                                   |
|  - n = length of x1 and x2;                                                                               |
|  - A, B, C, D = function paramters;                                                                       |
|  - director = defines the output of this function;                                                        |
|                                                                                                           |
|  OUTPUTS:                                                                                                 |
|  - y = A * x1 .* x2 + B * x1 + C * x2 + D                                                                 |
|  - return = (director = 1) inner product of x1 and x2                                                     |
|             (director = 2) reversed inner product of x1 and x2                                            |
|             (director = 3) Euclidean norm of y                                                            |
|                                                                                                           |
|  EXAMPLES:                                                                                                |
|  (a) Scalar multiplication:                                                                               |
|      none = vec_fun(u, v, w, N, 0, C, 0, 0, 0);  \\ w = C * u                                             |
|  (b) Component-wise addition:                                                                             |
|      none = vec_fun(u, v, w, N, 0, 1, 1, 0, 0);  \\ w = u .+ v                                            |
|  (c) Linear function:                                                                                     |
|      none = vec_fun(u, v, w, N, 0, C, 0, D, 0);  \\ w = C * u + D                                         |
|  (d) Component-wise multiplication:                                                                       |
|      none = vec_fun(u, v, w, N, 1, 0, 0, 0, 0);  \\ w = u .* v                                            |
|  (e) Inner product:                                                                                       |
|      inner_product = vec_fun(u, v, w, N, 1, 0, 0, 0, 1);  \\ inner product of u & v                       |
|  (f) Euclidean norm:                                                                                      |
|      euc_norm = vec_fun(u, v, w, N, 0, 1, 1, 0, 3);  \\ Euclidean norm of w                               |
\*---------------------------------------------------------------------------------------------------------*/
float vec_fun(float x1[], float x2[], float y[], int n, float A, float B, float C, float D, int director)
{
	/*
	y = A * x1 .* x2 + B * x1 + C * x2 + D
	*/
	float sum = 0.0f;
	for (int i = 0; i < n; i++)
	{
		y[i] = A * x1[i] * x2[i] + B * x1[i] + C * x2[i] + D;
	}
	switch (director)
	{
	case 1:
	{
		for (int i = 0; i < n; i++) { sum = sum + x1[i] * x2[i]; }
		break;
	}
	case 2:
	{
		for (int i = 0; i < n; i++) { sum = sum + x1[i] * x2[n - i - 1]; }
		break;
	}
	case 3:
	{
		for (int i = 0; i < n; i++) { sum = sum + y[i] * y[i]; }
		break;
		sum = sqrt(sum);
	}
	default: break;
	}
	return sum;
}

int main(void)
{
	/*display*/ printf("\n Homework 1, ME 599 H, WI 2017 \n Name: JIN Hongxi \n Student ID: 1628292 \n UW NetID: jinhx \n");
	/*display*/ printf("\n Input: \n");

	const int N = 5;
	/*display*/ printf("   N = %i; \n", N);

	float u[N], v[N], w[N];
	float K = -1.0f;

	for (int i = 0; i < N; i++)
	{
		u[i] = 1.0f / (N - 1);
		v[i] = 1.0f;
	}
	u[0] = 1.0f;
	/*useless*/ printf("   u = [ %.2f , %.2f , %.2f , %.2f , %.2f ]; \n", u[0], u[1], u[2], u[3], u[4]);
	/*useless*/ printf("   v = [ %.2f , %.2f , %.2f , %.2f , %.2f ]; \n", v[0], v[1], v[2], v[3], v[4]);

	/*useless*/ printf("\n Output: \n");

	float fout = vec_fun(u, v, w, N, 0, K, 0, 0, 0);
	/*useless*/ printf("   z = - u = [ %.2f , %.2f , %.2f , %.2f , %.2f ]; \n", w[0], w[1], w[2], w[3], w[4]);

	fout = vec_fun(u, w, w, N, 0, 1, 1, 0, 3);
	/*useless*/ printf("   u + z = [ %.2f , %.2f , %.2f , %.2f , %.2f ]; \n", w[0], w[1], w[2], w[3], w[4]);
	/*useless*/ printf("   The norm of u + z = %f; \n", fout);

	fout = vec_fun(u, v, w, N, 0, 0, 0, 0, 1);
	/*useless*/ printf("   The inner product of u and v = %f; \n", fout);

	fout = vec_fun(u, v, w, N, 0, 0, 0, 0, 2);
	/*useless*/ printf("   The reversed inner product of u and v = %f; \n", fout);

	/*useless*/ printf("\n Output: \n");

	float stranger;
	int top = 20000000, bottom = N, M;
	while ((top - bottom) > 1)
	{
		stranger = 1.0f * 1.0f;
		M = (top + bottom) / 2;
		for (int i = 1; i < M; i++) 
		{ 
			stranger = stranger + 1.0f / (M - 1); 
		}
		if (stranger < 2.0f) 
		{ 
			top = M; 
		}
		else 
		{ 
			bottom = M; 
		}
	}
	if (stranger < 2.0f) \
	{
		/*useless*/ printf("   The inner product of order %i = %f; \n", M--, stranger);
		for (int i = 1; i < M; i++) { stranger = stranger + 1.0f / (M - 1); }
		/*useless*/ printf("   The inner product of order %i = %f; \n", M, stranger);
	}
	else
	{
		/*useless*/ printf("   The inner product of order %i = %f; \n", M++, stranger);
		for (int i = 1; i < M; i++) { stranger = stranger + 1.0f / (M - 1); }
		/*useless*/ printf("   The inner product of order %i = %f; \n", M, stranger);
	}

	/*useless*/ printf("\n Press any key to exit. \n");
	/*useless*/ getchar();
	return 0;
}
/*---------------------------------------------------------------------------------------------------------*\
|  PART 2: main func                                                                                        |
|                                                                                                           |
|    a. INPUTS:                                                                                             |
|       N = 5;                                                                                              |
|       u = [ 1.00 , 0.25 , 0.25 , 0.25 , 0.25 ];                                                           |
|       v = [ 1.00 , 1.00 , 1.00 , 1.00 , 1.00 ];                                                           |
|  iii) z = [ -1.00 , -0.25 , -0.25 , -0.25 , -0.25 ];                                                      |
|       u + z = [ 0.00 , 0.00 , 0.00 , 0.00 , 0.00 ];                                                       |
|       Norm of ( u + z ) = 0.000000;                                                                       |
|   iv) Inner prod of ( u & v ) = 2.000000;                                                                 |
|    v) Reversed inner prod of ( u & v ) = 2.000000;                                                        |
|                                                                                                           |
|    b. Between N = 16777216 and 16777217, the dot product jumps from 2.000000 to 1.000000 suddenly.        |
|       This is probably caused by the inaccuracy of the program when assigning values to floating-points,  |
|       e.g. 1.5f cannot be exactly 1.500000.  Thus, when the order of inner product increases, the error   |
|       accummulates and will eventually give a "wrong" result at some point.  In this computer, where the  |
|       size of a floating-point is 4, is 16777216.                                                         |
\*---------------------------------------------------------------------------------------------------------*/