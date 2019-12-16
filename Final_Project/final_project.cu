#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "dev_array.h"
#include <math.h>
#include <random>
#include <algorithm>
#include<string>
#include <fstream>
#include <sstream>
using namespace std;


vector<string> split(const string &s, char delim) {
	stringstream ss(s);
	string item;
	vector<string> tokens;
	while (getline(ss, item, delim)) {
		tokens.push_back(item);
	}
	return tokens;
}

vector <float> operator/(const vector <float>& m2, const float m1) {

	/*  Returns the product of a float and a vectors (elementwise multiplication).
	 Inputs:
	 m1: float
	 m2: vector
	 Output: vector, m1 * m2, product of two vectors m1 and m2
	 */

	const unsigned long VECTOR_SIZE = m2.size();
	vector <float> product(VECTOR_SIZE);

	for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
		product[i] = m2[i] / m1;
	};

	return product;
}
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
	if (Col_Size == 10) {
		dim3 threadsPerBlock(10, 16);
		dim3 blocksPerGrid(Col_Size / 10, Row_Size / 16);
		matrixMultiplicationKernel << <blocksPerGrid, threadsPerBlock >> > (A, B, C, K_Width, Col_Size);
	}
	else {
		dim3 threadsPerBlock(16, 16);
		dim3 blocksPerGrid(Col_Size / 16, Row_Size / 16);
		matrixMultiplicationKernel << <blocksPerGrid, threadsPerBlock >> > (A, B, C, K_Width, Col_Size);
	}
}

void relu(float *Input, float *Output, int size) {
	reluKernel <<< size/16, 16>> > (Input,Output);
}

float* reluPrime(float *Input, int size) {
	float *Output = NULL;
	cudaMalloc(&Output, size * sizeof(float));
	reluPrimeKernel<<< size/16, 16>>>(Input,Output);
	return Output;
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
	if (Col_Size == 10) {
		dim3 threadsPerBlock(10, 16);
		dim3 blocksPerGrid(Col_Size / 10, Row_Size / 16);
		matrixMinusKernel << < blocksPerGrid, threadsPerBlock >> > (A, B, C, Col_Size);
	}
	else {
		dim3 threadsPerBlock(16, 16);
		dim3 blocksPerGrid(Col_Size / 16, Row_Size / 16);
		matrixMinusKernel << < blocksPerGrid, threadsPerBlock >> > (A, B, C, Col_Size);
	}
}

void matrixProduct(float *A, float *B, float *C, int Row_Size, int Col_Size) {
	dim3 threadsPerBlock(16,16);
	dim3 blocksPerGrid(Col_Size / 16, Row_Size / 16);
	matrixProductKernel << < blocksPerGrid, threadsPerBlock>> > (A,B,C,Col_Size);
}

float* matrixValueProduct(float *Input, int size, float value) {
	float *Output = NULL;
	cudaMalloc(&Output, size*sizeof(float));
	matrixValueProductKernel<< < size / 16, 16 >> > (Input, Output, value);
	return Output;
}

void matrixValueDivide(float *Input, float *Output, int size, float value) {
	matrixValueDivideKernel << < size / 16, 16 >> > (Input, Output, value);
}

void matrixTranspose(float *Input, float *Output, int Row_Size, int Col_Size) {
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(Col_Size / 16, Row_Size / 16);
	matrixTransposeKernel << < blocksPerGrid, threadsPerBlock >> > (Input,Output,Row_Size,Col_Size);
}

float* matrixTranspose_secondv(float *Input, int Row_Size, int Col_Size) {
	if (Col_Size == 10) {
		dim3 threadsPerBlock(10, 16);
		dim3 blocksPerGrid(Col_Size / 10, Row_Size / 16);
		float *Output = NULL;
		cudaMalloc(&Output, Row_Size * Col_Size * sizeof(float));
		matrixTransposeKernel << < blocksPerGrid, threadsPerBlock >> > (Input, Output, Row_Size, Col_Size);
		return Output;
	}
	else {
		dim3 threadsPerBlock(16, 16);
		dim3 blocksPerGrid(Col_Size / 16, Row_Size / 16);
		float *Output = NULL;
		cudaMalloc(&Output, Row_Size * Col_Size * sizeof(float));
		matrixTransposeKernel << < blocksPerGrid, threadsPerBlock >> > (Input, Output, Row_Size, Col_Size);
		return Output;
	}
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

void print_value(float *M, int Row_Size, int Col_Size) {
	for (int i = 0; i < Row_Size; i++) {
		for (int j = 0; j < Col_Size; j++) {
			printf("%f ",M[i*Col_Size + j]);
		}
		printf("\n\n");
	}
}

float compute_accuracy(float *prediction,float *ground_truth, int Row_Size, int Col_Size) {
	float correct = 0;
	for (int i = 0;i < Row_Size;i++) {
		int index1 = distance(&prediction[i*Col_Size], max_element(&prediction[i*Col_Size], &prediction[i*Col_Size] + Col_Size));
		int index2 = distance(&ground_truth[i*Col_Size], max_element(&ground_truth[i*Col_Size], &ground_truth[i*Col_Size] + Col_Size));
		if (index1 == index2)
			correct += 1;
	}
	return correct / Row_Size;
}
int main(int argc, char *argv[]) {
	// generate W1,W2,W3
	//matrixMultiplication
	string line;
	vector<string> line_v;
	cout << "Loading data ...\n";
	vector<float> X_train;
	vector<float> y_train;
	ifstream myfile("./train.txt");
	if (myfile.is_open()) {
		while (getline(myfile, line)) {
			line_v = split(line, '\t');
			int digit = strtof((line_v[0]).c_str(), 0);
			for (unsigned i = 0; i < 10; ++i) {
				if (i == digit)
				{
					y_train.push_back(1.);
				}
				else y_train.push_back(0.);
			}
			int size = static_cast<int>(line_v.size());
			for (unsigned i = 1; i < size; ++i) {
				X_train.push_back(strtof((line_v[i]).c_str(), 0));
			}
		}
		X_train = X_train / 255.0;
	}
	else cout << "Unable to open file" << '\n';
	cout << X_train.size();
	myfile.close();


	int BATCH_SIZE = 256;
	float lr = .01 / BATCH_SIZE;

	// Random initialization of the weights
	float *W1 = NULL, *W2 = NULL, *W3 = NULL, *b_x = NULL, *b_y = NULL;
	float *a1 = NULL, *a2 = NULL, *yhat = NULL, *dyhat = NULL;
	float *dw3 = NULL, *dz2 = NULL , *dw2 = NULL, *dz1 = NULL, *dw1=NULL;
	float *temp;

	// forward variable
	W1  = random_generate(W1, 784*128);
	W2  = random_generate(W2, 128*64);
	W3  = random_generate(W3, 64*10);
	b_x = (float *)malloc(BATCH_SIZE * 784 * sizeof(float));
	b_y = (float *)malloc(BATCH_SIZE * 10  * sizeof(float));
	a1 =  (float *)malloc(BATCH_SIZE * 128 * sizeof(float));
	a2 =  (float *)malloc(BATCH_SIZE * 64 * sizeof(float));
	yhat = (float *)malloc(BATCH_SIZE * 10 * sizeof(float));
	// 
	dyhat = (float *)malloc(BATCH_SIZE * 10 * sizeof(float));
	dw3   = (float *)malloc(64 * 10 * sizeof(float));
	dz2   = (float *)malloc(256 * 64 * sizeof(float));
	dw2   = (float *)malloc(128 * 64 * sizeof(float));
	dz1   = (float *)malloc(256 * 128 * sizeof(float));
	dw1   = (float *)malloc(784 * 128 * sizeof(float));

	dev_array W1_d(784, 128);
	dev_array W2_d(128, 64);
	dev_array W3_d(64, 10);
	dev_array b_x_d(BATCH_SIZE , 784);
	dev_array b_y_d(BATCH_SIZE , 10);
	dev_array a1_d(BATCH_SIZE, 128);
	dev_array a2_d(BATCH_SIZE, 64);
	dev_array yhat_d(BATCH_SIZE, 10);
	dev_array dyhat_d(BATCH_SIZE, 10);
	dev_array dw3_d(64, 10);
	dev_array dz2_d(256, 64);
	dev_array dw2_d(128 ,64);
	dev_array dz1_d(256, 128);
	dev_array dw1_d(784, 128);

	W1_d.set(W1);
	W2_d.set(W2);
	W3_d.set(W3);
	b_x_d.set(b_x);
	b_y_d.set(b_y);
	a1_d.set(a1);
	a2_d.set(a2);
	yhat_d.set(yhat);
	dyhat_d.set(dyhat);
	dw3_d.set(dw3);
	dz2_d.set(dz2);
	dw2_d.set(dw2);
	dz1_d.set(dz1);
	dw1_d.set(dw1);

	cout << "Training the model ...\n";

	for (unsigned i = 0 ; i < 32000; i++) {
		int randindx = rand() % (37904 - BATCH_SIZE);
		copy(X_train.begin()+ randindx * 784, X_train.begin() + (randindx+ BATCH_SIZE)*784, b_x);
		cudaMemcpy(b_x_d.getData(), b_x,  BATCH_SIZE * 784 * sizeof(float) ,cudaMemcpyHostToDevice);
		
		copy(y_train.begin() + randindx * 10, y_train.begin() + (randindx + BATCH_SIZE) * 10, b_y);
		cudaMemcpy(b_y_d.getData(), b_y, BATCH_SIZE * 10 * sizeof(float), cudaMemcpyHostToDevice);
	    
		matrixMultiplication(b_x_d.getData(), W1_d.getData(), a1_d.getData(), b_x_d.getRowSize(), W1_d.getColSize(), b_x_d.getColSize());
		relu(a1_d.getData(), a1_d.getData(), 256 * 128);

		matrixMultiplication(a1_d.getData(), W2_d.getData(), a2_d.getData(), 256, 64, 128);
		relu(a2_d.getData(), a2_d.getData(), 256 * 64);

		matrixMultiplication(a2_d.getData(), W3_d.getData(), yhat_d.getData(), 256, 10, 64);
		softmax(yhat_d.getData(), yhat_d.getData(), 256*10);
		//yhat_d.get(yhat);

		// Back propagation
		matrixMinus(yhat_d.getData(), b_y_d.getData(), dyhat_d.getData(), 256, 10);

		temp = matrixTranspose_secondv(a2_d.getData(), BATCH_SIZE, 64);
		matrixMultiplication(temp, dyhat_d.getData(), dw3_d.getData(), 64, 10, 256);
		cudaFree(temp);

		temp = matrixTranspose_secondv(W3_d.getData(), 64, 10);
		matrixMultiplication(dyhat_d.getData(), temp, dz2_d.getData(),256,64,10);
		cudaFree(temp);

		temp = reluPrime(a2_d.getData(), a2_d.getSize());
		matrixProduct(dz2_d.getData(), reluPrime(a2_d.getData(),a2_d.getSize()) , dz2_d.getData(), 256, 64);
		cudaFree(temp);

		temp = matrixTranspose_secondv(W2_d.getData(), 128, 64);
		matrixMultiplication(dz2_d.getData(), temp, dz1_d.getData(), 256,128,64);
		cudaFree(temp);

		temp = reluPrime(a1_d.getData(), a1_d.getSize());
		matrixProduct(dz1_d.getData(), reluPrime(a1_d.getData(), a1_d.getSize()), dz1_d.getData(), 256,128);
		cudaFree(temp);

		temp = matrixTranspose_secondv(b_x_d.getData(), 256, 784);
		matrixMultiplication(temp,dz1_d.getData(), dw1_d.getData(),784,128,256);
		cudaFree(temp);

		temp = matrixValueProduct(dw3_d.getData(), dw3_d.getSize(), lr);
		matrixMinus(W3_d.getData(), temp, W3_d.getData(), 64, 10);
		cudaFree(temp);

		temp = matrixValueProduct(dw2_d.getData(), dw2_d.getSize(), lr);
		matrixMinus(W2_d.getData(), temp, W2_d.getData(), 128, 64);
		cudaFree(temp);

		temp = matrixValueProduct(dw1_d.getData(), dw1_d.getSize(), lr);
		matrixMinus(W1_d.getData(), temp, W1_d.getData(), 784, 128);
		cudaFree(temp);
		if ((i + 1) % 100 == 0) {
			cout << "-----------------------------------------------Epoch " << i + 1 << "--------------------------------------------------" << "\n";
			//cout << "Predictions:" << "\n";
			yhat_d.get(yhat);
			//print_value(yhat, 10, 10);
			//cout << "Ground truth:" << "\n";
			b_y_d.get(b_y);
			//print_value(b_y, 10, 10);
			cout << compute_accuracy(yhat, b_y, 256, 10) << endl;;
		}
	}
	cout << endl;
	cout << "Testing the model ...\n";
	// testing
	for (int i = 0; i < 16; i++) {
		int randindx = 37096 + i * 256;
		copy(X_train.begin() + randindx * 784, X_train.begin() + (randindx + BATCH_SIZE) * 784, b_x);
		cudaMemcpy(b_x_d.getData(), b_x, BATCH_SIZE * 784 * sizeof(float), cudaMemcpyHostToDevice);

		copy(y_train.begin() + randindx * 10, y_train.begin() + (randindx + BATCH_SIZE) * 10, b_y);
		cudaMemcpy(b_y_d.getData(), b_y, BATCH_SIZE * 10 * sizeof(float), cudaMemcpyHostToDevice);

		matrixMultiplication(b_x_d.getData(), W1_d.getData(), a1_d.getData(), b_x_d.getRowSize(), W1_d.getColSize(), b_x_d.getColSize());
		relu(a1_d.getData(), a1_d.getData(), 256 * 128);

		matrixMultiplication(a1_d.getData(), W2_d.getData(), a2_d.getData(), 256, 64, 128);
		relu(a2_d.getData(), a2_d.getData(), 256 * 64);

		matrixMultiplication(a2_d.getData(), W3_d.getData(), yhat_d.getData(), 256, 10, 64);
		softmax(yhat_d.getData(), yhat_d.getData(), 256 * 10);
		//yhat_d.get(yhat);

		// Back propagation
		matrixMinus(yhat_d.getData(), b_y_d.getData(), dyhat_d.getData(), 256, 10);

		temp = matrixTranspose_secondv(a2_d.getData(), BATCH_SIZE, 64);
		matrixMultiplication(temp, dyhat_d.getData(), dw3_d.getData(), 64, 10, 256);
		cudaFree(temp);

		temp = matrixTranspose_secondv(W3_d.getData(), 64, 10);
		matrixMultiplication(dyhat_d.getData(), temp, dz2_d.getData(), 256, 64, 10);
		cudaFree(temp);

		temp = reluPrime(a2_d.getData(), a2_d.getSize());
		matrixProduct(dz2_d.getData(), reluPrime(a2_d.getData(), a2_d.getSize()), dz2_d.getData(), 256, 64);
		cudaFree(temp);

		temp = matrixTranspose_secondv(W2_d.getData(), 128, 64);
		matrixMultiplication(dz2_d.getData(), temp, dz1_d.getData(), 256, 128, 64);
		cudaFree(temp);

		temp = reluPrime(a1_d.getData(), a1_d.getSize());
		matrixProduct(dz1_d.getData(), reluPrime(a1_d.getData(), a1_d.getSize()), dz1_d.getData(), 256, 128);
		cudaFree(temp);

		temp = matrixTranspose_secondv(b_x_d.getData(), 256, 784);
		matrixMultiplication(temp, dz1_d.getData(), dw1_d.getData(), 784, 128, 256);
		cudaFree(temp);

		temp = matrixValueProduct(dw3_d.getData(), dw3_d.getSize(), lr);
		matrixMinus(W3_d.getData(), temp, W3_d.getData(), 64, 10);
		cudaFree(temp);

		temp = matrixValueProduct(dw2_d.getData(), dw2_d.getSize(), lr);
		matrixMinus(W2_d.getData(), temp, W2_d.getData(), 128, 64);
		cudaFree(temp);

		temp = matrixValueProduct(dw1_d.getData(), dw1_d.getSize(), lr);
		matrixMinus(W1_d.getData(), temp, W1_d.getData(), 784, 128);
		cudaFree(temp);
		cout << "--------------------------------------------testing batch " << i + 1 << "--------------------------------------------------" << "\n";
		yhat_d.get(yhat);
		b_y_d.get(b_y);
		cout << compute_accuracy(yhat, b_y, 256, 10) << endl;;
	}

}
