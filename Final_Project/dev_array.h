#ifndef _DEV_ARRAY_H_
#define _DEV_ARRAY_H_

#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

class dev_array
{
public:
	explicit dev_array();
	explicit dev_array(int row_size, int col_size);
	~dev_array();

	void resize(int row_size, int col_size);

	size_t getSize() const;
	float *getData();
	int getRowSize() const;
	int getColSize() const;

	void set(const float* src, size_t size);
	void get(float* dest, size_t size);

private:
	void allocate(size_t size);
	void free();
	float* start_;
	float* end_;
	int row_size_;
	int col_size_;
};

#endif
