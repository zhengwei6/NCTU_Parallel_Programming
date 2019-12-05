#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "dev_array.h"
#include <math.h>
#include <random>
#include <algorithm>

using namespace std;

__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int K_Width, int  Col_Size) {

	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	float Pvalue = 0;
	for (int k = 0; k < K_Width; k++) {
		Pvalue += A[Row*K_Width + k] * B[k*Col_Size + Col];
	}
	C[Row*Col_Size + Col] = Pvalue;
}

void matrixMultiplication(float *A, float *B, float *C, int Row_Size, int Col_Size, int K_Width) {
	dim3 threadsPerBlock(32, 32);
	dim3 blocksPerGrid(Row_Size / 32, Col_Size / 32);
	matrixMultiplicationKernel << <blocksPerGrid, threadsPerBlock >> > (A, B, C, K_Width, Col_Size);
}

float* random_generate(float *M, size_t size) {
	M = (float *)malloc(size * sizeof(float));
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> distribution(0.0, 0.05);
	static default_random_engine generator;
	generate(M, M + size, [&]() { return distribution(generator); });
	return M;
}

int main(int argc, char *argv[]) {

	// generate W1,W2,W3
	float *W1=NULL,*W2=NULL,*W3=NULL;
	W1 = random_generate(W1, 64 * 32);
	W2 = random_generate(W2, 32 * 64);
	W3 = (float *)malloc(64*64 * sizeof(float));
	dev_array W1_d(64, 32);
	dev_array W2_d(32, 64);
	dev_array W3_d(64, 64);
	W1_d.set(W1);
	W2_d.set(W2);
	W3_d.set(W3);
	matrixMultiplication(W1_d.getData(),W2_d.getData(), W3_d.getData(), W1_d.getRowSize() , W2_d.getColSize(), W1_d.getColSize());
	W3_d.get(W3, 64 * 64);
	for (int i = 0;i < 64;i++) {
		for (int j = 0;j < 64;j++)
			printf("%f ", W3[i * 64 + j]);
	}

}