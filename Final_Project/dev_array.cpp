#include "dev_array.h"

dev_array::dev_array() : start_(0), end_(0), row_size_(0), col_size_(0)
{}

dev_array::dev_array(int row_size, int col_size)
{
	size_t size = row_size * col_size;
	allocate(size);
	row_size_ = row_size;
	col_size_ = col_size;
}


dev_array::~dev_array()
{
	free();
}

void dev_array::resize(int row_size, int col_size)
{
	free();
	size_t size = row_size * col_size;
	row_size_ = row_size;
	col_size_ = col_size;
	allocate(size);
}

size_t dev_array::getSize() const
{
	return end_ - start_;
}

float* dev_array::getData()
{
	return start_;
}

int dev_array::getRowSize() const
{
	return row_size_;
}

int dev_array::getColSize() const
{
	return col_size_;
}

void dev_array::set(const float* src)
{
	cudaError_t result = cudaMemcpy(start_, src, getSize() * sizeof(float), cudaMemcpyHostToDevice);
	if (result != cudaSuccess)
	{
		throw std::runtime_error("failed to copy to device memory");
	}
}

void dev_array::get(float* dest, size_t size)
{
	size_t min = std::min(size, getSize());
	cudaError_t result = cudaMemcpy(dest, start_, min * sizeof(float), cudaMemcpyDeviceToHost);
	if (result != cudaSuccess)
	{
		throw std::runtime_error("failed to copy to host memory");
	}
}

void dev_array::allocate(size_t size)
{
	cudaError_t result = cudaMalloc((void**)&start_, size * sizeof(float));
	if (result != cudaSuccess)
	{
		start_ = end_ = 0;
		throw std::runtime_error("failed to allocate device memory");
	}
	end_ = start_ + size;
}

void dev_array::free()
{
	if (start_ != 0)
	{
		cudaFree(start_);
		start_ = end_ = 0;
		col_size_ = row_size_ = 0;
	}
}