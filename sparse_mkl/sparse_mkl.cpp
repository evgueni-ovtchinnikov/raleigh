// sparse_mkl.cpp : Defines the exported functions for the DLL application.
//
#include "stdafx.h"

#include <mkl.h>

#ifdef _WIN32
#define EXPORTED_FUNCTION __declspec(dllexport)
#else
#define EXPORTED_FUNCTION
#endif

extern "C"
EXPORTED_FUNCTION int num_mkl_threads()
{
	return mkl_get_max_threads();
}

extern "C"
EXPORTED_FUNCTION void set_num_mkl_threads(int t)
{
	return mkl_set_num_threads(t);
}

extern "C"
EXPORTED_FUNCTION void scsrmv
(int m, int* ia, int* ja, float* a, float* u, float* v)
{
	mkl_scsrgemv("N", &m, a, ia, ja, u, v);
}

extern "C"
EXPORTED_FUNCTION void dcsrmv
(int m, int* ia, int* ja, double* a, double* u, double* v)
{
	mkl_dcsrgemv("N", &m, a, ia, ja, u, v);
}

extern "C"
EXPORTED_FUNCTION void scsrmm
(int m, int n, int k, int* ia, int* ja, float* a, float* u, float* v)
{
	float alpha = 1.0f;
	float beta = 0.0f;

	//for (int j = 0; j < k; j++) {
	//	std::cout << "col " << j << '\n';
	//	for (int i = 0; i < n; i++) {
	//		std::cout << u[i + j*n] << '\n';
	//	}
	//}
	int* pb = ia;
	int* pe = ia + 1;
	//for (int i = 0; i < m; i++) {
	//	std::cout << "row " << i << '\n';
	//	for (int j = pb[i]; j < pe[i]; j++)
	//		std::cout << ja[j - 1] - 1 << '\t' << a[j - 1] << '\n';
	//}
	mkl_scsrmm("N", &m, &k, &n, &alpha, "G", a, ja, pb, pe, u, &n, &beta, v, &m);
	//for (int j = 0; j < k; j++) {
	//	std::cout << "col " << j << '\n';
	//	for (int i = 0; i < m; i++) {
	//		std::cout << v[i + j*m] << '\n';
	//	}
	//}
}

extern "C"
EXPORTED_FUNCTION void dcsrmm
(int m, int n, int k, int* ia, int* ja, double* a, double* u, double* v)
{
	double alpha = 1.0;
	double beta = 0.0;

	int* pb = ia;
	int* pe = ia + 1;
	mkl_dcsrmm("N", &m, &k, &n, &alpha, "G", a, ja, pb, pe, u, &n, &beta, v, &m);
}


