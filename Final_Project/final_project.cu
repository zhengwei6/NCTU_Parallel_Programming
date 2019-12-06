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

__global__ void reluKernel(float *Input, float *Output) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (Input[index] < 0) {
		Output[index] = 0.0;
	}
	else {
		Output[index] = Input[index];
	}
}

__global__ void reluPrimeKernel(float *Input, float *Output) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (Input[index] <= 0) {
		Output[index] = 0.0;
	}
	else {
		Output[index] = 1.0;
	}
}

__global__ void sigmoidKernel(float *Input, float *Output) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	Output[index] = 1 / (1 + expf(-Input[index]));
}

__global__ void sigmoid_dKernel(float *Input, float *Output) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	Output[index] = Input[index] * (1 - Input[index]);
}

__global__ void matrixAddKernel(float *A, float *B, float *C, int Col_Size) {
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	C[Row*Col_Size + Col] = A[Row*Col_Size + Col] + B[Row*Col_Size + Col];
}

__global__ void matrixMinusKernel(float *A, float *B, float *C, int Col_Size) {
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	C[Row*Col_Size + Col] = A[Row*Col_Size + Col] - B[Row*Col_Size + Col];
}

__global__ void matrixProductKernel(float *A, float *B, float *C, int Col_Size) {
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	C[Row*Col_Size + Col] = A[Row*Col_Size + Col] * B[Row*Col_Size + Col];
}

__global__ void matrixValueProductKernel(float *Input, float *Output, float value) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	Output[index] = Input[index] * value;
}

__global__ void matrixValueDivideKernel(float *Input, float *Output, float value) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	Output[index] = Input[index] / value;
}

__global__ void matrixTransposeKernel(float *Input, float *Output, int Row_Size, int Col_Size) {
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	Output[Col*Row_Size + Row] = Input[Row*Col_Size + Col];
}

__global__ void sofmaxKernel(float *Input, float *Output) {
	int index = threadIdx.x;
	float foo[10];
	float max = Input[index * 10];
	for (int i = 0; i < 10; i++) {
		foo[i] = Input[index * 10 + i];
		if (foo[i] > max) {
			max = foo[i];
		}
	}
	for (int i = 0;i<10; i++) {
		foo[i] = expf(foo[i] - max);
	}
	float sum = 0.0;
	for (int j = 0; j < 10;j++) {
		sum = sum + foo[j];
	}
	for (int j = 0;j < 10; j++) {
		Output[index * 10 + j] = foo[j] / sum;
	}
}
/////////////////////////////////////////////////////////////////////////////////
void matrixMultiplication(float *A, float *B, float *C, int Row_Size, int Col_Size, int K_Width) {
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(Col_Size / 16,Row_Size / 16);
	matrixMultiplicationKernel << <blocksPerGrid, threadsPerBlock >> > (A, B, C, K_Width, Col_Size);
}

void relu(float *Input, float *Output, int size) {
	reluKernel <<< size/16, 16>> > (Input,Output);
}

void reluPrime(float *Input, float *Output, int size) {
	reluPrimeKernel<<< size/16, 16>>>(Input,Output);
}

void sigmoid(float *Input, float *Output, int size) {
	sigmoidKernel << < size / 16, 16 >> > (Input,Output);
}

void sigmoid_d(float *Input, float *Output, int size) {
	sigmoid_dKernel<<< size / 16, 16 >> > (Input, Output);
}

void matrixAdd(float *A, float *B, float *C, int Row_Size, int Col_Size) {
	dim3 threadsPerBlock(16,16);
	dim3 blocksPerGrid(Col_Size / 16,Row_Size / 16);
	matrixAddKernel << < blocksPerGrid, threadsPerBlock >> > (A, B, C, Col_Size);
}

void matrixMinus(float *A, float *B, float *C, int Row_Size, int Col_Size) {
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(Col_Size/16, Row_Size/16);
	matrixMinusKernel << < blocksPerGrid, threadsPerBlock >> > (A,B,C,Col_Size);
}

void matrixProduct(float *A, float *B, float *C, int Row_Size, int Col_Size) {
	dim3 threadsPerBlock(16,16);
	dim3 blocksPerGrid(Col_Size / 16, Row_Size / 16);
	matrixProductKernel << < blocksPerGrid, threadsPerBlock>> > (A,B,C,Col_Size);
}

void matrixValueProduct(float *Input, float *Output, int size, float value) {
	matrixValueProductKernel<< < size / 16, 16 >> > (Input, Output, value);
}

void matrixValueDivide(float *Input, float *Output, int size, float value) {
	matrixValueDivideKernel << < size / 16, 16 >> > (Input, Output, value);
}

void matrixTranspose(float *Input, float *Output, int Row_Size, int Col_Size) {
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(Col_Size / 16, Row_Size / 16);
	matrixTransposeKernel << < blocksPerGrid, threadsPerBlock >> > (Input,Output,Row_Size,Col_Size);
}

void softmax(float *Input, float *Output,int size) {
	sofmaxKernel << <1,size / 10 >> > (Input, Output);
}

/////////////////////////////////////////////////////////////////////
float* test_generate(float *M, size_t size, float num) {
	M = (float *)malloc(size * sizeof(float));
	for (int i = 0; i < size; i++) {
		M[i] = num;
	}
	return M;
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
	/*
	//matrixMultiplication
	float *W1=NULL,*W2=NULL,*W3=NULL;
	W1 = test_generate(W1, 64 * 32,1);
	W2 = test_generate(W2, 32 * 64,1);
	W3 = (float *)malloc(64*64 * sizeof(float));
	dev_array W1_d(64, 32);
	dev_array W2_d(32, 64);
	dev_array W3_d(64, 64);
	W1_d.set(W1);
	W2_d.set(W2);
	W3_d.set(W3);
	matrixMultiplication(W1_d.getData(),W2_d.getData(), W3_d.getData(), W1_d.getRowSize() , W2_d.getColSize(), W1_d.getColSize());
	W3_d.get(W3);
	for (int i = 0;i < 64;i++) {
		for (int j = 0;j < 64;j++)
			printf("%f ", W3[i * 64 + j]);
	}
	*/
	/*
	float *W1 = NULL,*W3=NULL;
	W1 = random_generate(W1, 128*1);
	W3 = (float *)malloc(128 * 1 * sizeof(float));
	dev_array W1_d(128, 1);
	dev_array W3_d(128, 1);
	W1_d.set(W1);
	W3_d.set(W3);
	relu(W1_d.getData(), W3_d.getData(), W3_d.getSize());
	W3_d.get(W3);
	for (int i = 0;i < 128;i++) {
			printf("%f ", W3[i]);
	}
	*/
	/*
	float *W1 = NULL,*W3=NULL;
	W1 = test_generate(W1, 128*1,1);
	W3 = (float *)malloc(128 * 1 * sizeof(float));
	dev_array W1_d(128, 1);
	dev_array W3_d(128, 1);
	W1_d.set(W1);
	W3_d.set(W3);
	sigmoid(W1_d.getData(), W3_d.getData(), W3_d.getSize());
	W3_d.get(W3);
	for (int i = 0;i < 128;i++) {
			printf("%f ", W3[i]);
	}
	*/
	/*
	float *W1 = NULL,*W2 = NULL,*W3=NULL;
	W1 = test_generate(W1, 128*16,1);
	W2 = test_generate(W2, 1280*16,1);
	W3 = (float *)malloc(128 * 16 * sizeof(float));
	dev_array W1_d(128, 16);
	dev_array W2_d(128, 16);
	dev_array W3_d(128, 16);
	W1_d.set(W1);
	W2_d.set(W1);
	W3_d.set(W3);
	matrixAdd(W1_d.getData(), W2_d.getData(), W3_d.getData(), 128, 16);
	W3_d.get(W3);
	for (int i = 0;i < 128*16;i++) {
			printf("%f ", W3[i]);
	}
	*/
	/*
	float *W1 = NULL,*W2 = NULL,*W3=NULL;
	W1 = test_generate(W1, 128*16,1);
	W2 = test_generate(W2, 1280*16,1);
	W3 = (float *)malloc(128 * 16 * sizeof(float));
	dev_array W1_d(128, 16);
	dev_array W2_d(128, 16);
	dev_array W3_d(128, 16);
	W1_d.set(W1);
	W2_d.set(W1);
	W3_d.set(W3);
	matrixMinus(W1_d.getData(), W2_d.getData(), W3_d.getData(), 128, 16);
	W3_d.get(W3);
	for (int i = 0;i < 128*16;i++) {
			printf("%f ", W3[i]);
	}
	*/
	/*
	float *W1 = NULL,*W2 = NULL,*W3=NULL;
	W1 = test_generate(W1, 128*16,1);
	W2 = test_generate(W2, 128*16,2);
	W3 = (float *)malloc(128 * 16 * sizeof(float));
	dev_array W1_d(128, 16);
	dev_array W2_d(128, 16);
	dev_array W3_d(128, 16);
	W1_d.set(W1);
	W2_d.set(W2);
	W3_d.set(W3);
	matrixProduct(W1_d.getData(), W2_d.getData(), W3_d.getData(), 128, 16);
	W3_d.get(W3);
	for (int i = 0;i < 128*16;i++) {
			printf("%f ", W3[i]);
	}
	*/
	/*
	float *W1 = NULL,*W3=NULL;
	W1 = test_generate(W1, 128*1,1);
	W3 = (float *)malloc(128 * 1 * sizeof(float));
	dev_array W1_d(128, 1);
	dev_array W3_d(128, 1);
	W1_d.set(W1);
	W3_d.set(W3);
	matrixValueProduct(W1_d.getData(), W1_d.getData(), W1_d.getSize(), 5);
	W1_d.get(W1);
	for (int i = 0;i < 128;i++) {
			printf("%f ", W1[i]);
	}
	*/
    /*
	float *W1 = NULL,*W3=NULL;
	W1 = test_generate(W1, 128*1,1.0);
	W3 = (float *)malloc(128 * 1 * sizeof(float));
	dev_array W1_d(128, 1);
	dev_array W3_d(128, 1);
	W1_d.set(W1);
	W3_d.set(W3);
	matrixValueDivide(W1_d.getData(), W3_d.getData(), W1_d.getSize(), 5);
	W3_d.get(W3);
	for (int i = 0;i < 128;i++) {
			printf("%f ", W3[i]);
	}
	*/
     /*
	float *W1 = NULL,*W3=NULL;
	W1 = test_generate(W1, 128*16,2.0);
	W3 = (float *)malloc(16*128 * sizeof(float));
	dev_array W1_d(128, 16);
	dev_array W3_d(16, 128);
	W1_d.set(W1);
	W3_d.set(W3);
	matrixTranspose(W1_d.getData(), W3_d.getData(), 128, 16);
	W3_d.get(W3);
	for (int i = 0;i < 16;i++) {
		for(int j=0;j < 128;j++)
			printf("%f ", W3[i*128+j]);
	}
	*/
	float *W1 = NULL,*W3=NULL;
	W1 = test_generate(W1, 256*10,2.0);
	W3 = (float *)malloc(256*10 * sizeof(float));
	dev_array W1_d(256, 10);
	dev_array W3_d(256, 10);
	W1_d.set(W1);
	W3_d.set(W3);
	softmax(W1_d.getData(), W3_d.getData(), W1_d.getSize());
	W3_d.get(W3);
	for (int i = 0;i < 256;i++) {
		for(int j=0;j < 10;j++)
			printf("%f ", W3[i*10+j]);
	}
}