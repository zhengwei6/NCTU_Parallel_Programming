#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
//#include "kernel.h"
//#include "kernel.cu"
#include "dev_array.h"
#include <math.h>
#include <random>

using namespace std;

float* random_generate(float *M, size_t size) {
	M = (float *)malloc(size * sizeof(float));
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> distribution(0.0, 0.05);
	static default_random_engine generator;
	generate(M, M+size, [&]() { return distribution(generator); });
	return M;
}
int main(int argc, char *argv[]) {

	// generate W1,W2,W3
	float *W1, *W2, *W3;
	W1 = random_generate(W1,748*128);


}