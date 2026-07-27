#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

using namespace std;

// DEFINE NUM_REPS

// DEFINE KERNEL PARAMETERS
#define VALUES_PER_THREAD 4

// DEFINE PRECISION

// DEFINE DEVICE

// DEFINE MIXED PARAMETERS
#define MEMORY_TARGET_GLOBAL 0
#define MEMORY_TARGET_L2 1
#define MEMORY_TARGET_SHARED 2

#define CUDA_CHECK(call)                                                                  \
	do {                                                                                    \
		cudaError_t error__ = (call);                                                        \
		if (error__ != cudaSuccess) {                                                        \
			cerr << "CUDA error: " << cudaGetErrorString(error__) << " (" << __FILE__ << ':' \
				 << __LINE__ << ')' << endl;                                                   \
			return 1;                                                                         \
		}                                                                                   \
	} while (0)

__device__ __forceinline__ PRECISION mixed_operation(PRECISION value) {
	// DEFINE OPERATION
}

#if MEMORY_TARGET != MEMORY_TARGET_SHARED
__global__ void benchmark_stream(const PRECISION *__restrict__ input,
								 PRECISION *__restrict__ output, uint64_t elements) {
	const uint64_t thread_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;

	for (uint64_t base = thread_id; base < elements; base += stride * VALUES_PER_THREAD) {
		PRECISION value0 = input[base];
		PRECISION value1 = input[base + stride];
		PRECISION value2 = input[base + 2 * stride];
		PRECISION value3 = input[base + 3 * stride];

#pragma unroll 4
		for (int operation = 0; operation < OP_COUNT; ++operation) {
			value0 = mixed_operation(value0);
			value1 = mixed_operation(value1);
			value2 = mixed_operation(value2);
			value3 = mixed_operation(value3);
		}

		output[base] = value0;
		output[base + stride] = value1;
		output[base + 2 * stride] = value2;
		output[base + 3 * stride] = value3;
	}
}
#else
__global__ void benchmark_shared(PRECISION *sink, int iterations) {
	__shared__ PRECISION storage[VALUES_PER_THREAD * THREADS_PER_BLOCK];
	volatile PRECISION *shared_values = storage;

	const int local0 = threadIdx.x;
	const int local1 = threadIdx.x + THREADS_PER_BLOCK;
	const int local2 = threadIdx.x + 2 * THREADS_PER_BLOCK;
	const int local3 = threadIdx.x + 3 * THREADS_PER_BLOCK;
	const uint64_t global_base =
		(static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * VALUES_PER_THREAD;

	storage[local0] = sink[global_base];
	storage[local1] = sink[global_base + 1];
	storage[local2] = sink[global_base + 2];
	storage[local3] = sink[global_base + 3];
	__syncthreads();

	for (int iteration = 0; iteration < iterations; ++iteration) {
		PRECISION value0 = shared_values[local0];
		PRECISION value1 = shared_values[local1];
		PRECISION value2 = shared_values[local2];
		PRECISION value3 = shared_values[local3];

#pragma unroll 4
		for (int operation = 0; operation < OP_COUNT; ++operation) {
			value0 = mixed_operation(value0);
			value1 = mixed_operation(value1);
			value2 = mixed_operation(value2);
			value3 = mixed_operation(value3);
		}

		shared_values[local0] = value0;
		shared_values[local1] = value1;
		shared_values[local2] = value2;
		shared_values[local3] = value3;
	}
	__syncthreads();

	sink[global_base] = storage[local0];
	sink[global_base + 1] = storage[local1];
	sink[global_base + 2] = storage[local2];
	sink[global_base + 3] = storage[local3];
}
#endif

static float median(vector<float> values) {
	sort(values.begin(), values.end());
	if (values.size() % 2 == 0)
		return (values[values.size() / 2] + values[values.size() / 2 - 1]) / 2.f;
	return values[values.size() / 2];
}

int main() {
	CUDA_CHECK(cudaSetDevice(DEVICE));

	cudaDeviceProp properties;
	CUDA_CHECK(cudaGetDeviceProperties(&properties, DEVICE));

	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	float milliseconds = 0.f;
	vector<float> time_series;
	int iterations = 1;
	double transferred_bytes = 0.0;
	double executed_flops = 0.0;
	double actual_working_set_mib = 0.0;

#if MEMORY_TARGET == MEMORY_TARGET_SHARED
	const uint64_t launch_blocks = NUM_BLOCKS;
	const uint64_t values_per_launch =
		launch_blocks * THREADS_PER_BLOCK * VALUES_PER_THREAD;
	PRECISION *sink = nullptr;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&sink), values_per_launch * sizeof(PRECISION)));
	CUDA_CHECK(cudaMemset(sink, 0, values_per_launch * sizeof(PRECISION)));

	while (milliseconds < 150.f) {
		CUDA_CHECK(cudaEventRecord(start));
		benchmark_shared<<<launch_blocks, THREADS_PER_BLOCK>>>(sink, iterations);
		CUDA_CHECK(cudaEventRecord(stop));
		CUDA_CHECK(cudaEventSynchronize(stop));
		CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
		if (milliseconds < 150.f) iterations *= 2;
	}

	for (int repetition = 0; repetition < NUM_REPS; ++repetition) {
		CUDA_CHECK(cudaEventRecord(start));
		benchmark_shared<<<launch_blocks, THREADS_PER_BLOCK>>>(sink, iterations);
		CUDA_CHECK(cudaEventRecord(stop));
		CUDA_CHECK(cudaEventSynchronize(stop));
		CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
		time_series.push_back(milliseconds);
	}

	transferred_bytes = static_cast<double>(values_per_launch) * iterations *
		2.0 * sizeof(PRECISION);
	executed_flops = static_cast<double>(values_per_launch) * iterations *
		OP_COUNT * FLOPS_PER_ELEMENT_OP;
	actual_working_set_mib = static_cast<double>(VALUES_PER_THREAD * THREADS_PER_BLOCK *
		 sizeof(PRECISION)) / (1024.0 * 1024.0);
	CUDA_CHECK(cudaFree(sink));
#else
	uint64_t requested_elements = 0;
#if MEMORY_TARGET == MEMORY_TARGET_L2
	// Two buffers together occupy half of L2, leaving room for other cache traffic.
	requested_elements = static_cast<uint64_t>(properties.l2CacheSize) /
		(4ULL * sizeof(PRECISION));
#else
	requested_elements = REQUESTED_WORKING_SET_BYTES / (2ULL * sizeof(PRECISION));
#endif

	const uint64_t values_per_block = THREADS_PER_BLOCK * VALUES_PER_THREAD;
	uint64_t launch_blocks = requested_elements / values_per_block;
	launch_blocks = max<uint64_t>(1, min<uint64_t>(NUM_BLOCKS, launch_blocks));
	const uint64_t group_size = launch_blocks * values_per_block;
	uint64_t elements = (requested_elements / group_size) * group_size;
	if (elements == 0) elements = group_size;

	PRECISION *buffer_a = nullptr;
	PRECISION *buffer_b = nullptr;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffer_a), elements * sizeof(PRECISION)));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffer_b), elements * sizeof(PRECISION)));
	CUDA_CHECK(cudaMemset(buffer_a, 0, elements * sizeof(PRECISION)));
	CUDA_CHECK(cudaMemset(buffer_b, 0, elements * sizeof(PRECISION)));

	while (milliseconds < 150.f) {
		PRECISION *input = buffer_a;
		PRECISION *output = buffer_b;
		CUDA_CHECK(cudaEventRecord(start));
		for (int iteration = 0; iteration < iterations; ++iteration) {
			benchmark_stream<<<launch_blocks, THREADS_PER_BLOCK>>>(input, output, elements);
			swap(input, output);
		}
		CUDA_CHECK(cudaEventRecord(stop));
		CUDA_CHECK(cudaEventSynchronize(stop));
		CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
		if (milliseconds < 150.f) iterations *= 2;
	}

	for (int repetition = 0; repetition < NUM_REPS; ++repetition) {
		PRECISION *input = buffer_a;
		PRECISION *output = buffer_b;
		CUDA_CHECK(cudaEventRecord(start));
		for (int iteration = 0; iteration < iterations; ++iteration) {
			benchmark_stream<<<launch_blocks, THREADS_PER_BLOCK>>>(input, output, elements);
			swap(input, output);
		}
		CUDA_CHECK(cudaEventRecord(stop));
		CUDA_CHECK(cudaEventSynchronize(stop));
		CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
		time_series.push_back(milliseconds);
	}

	transferred_bytes = static_cast<double>(elements) * iterations * 2.0 * sizeof(PRECISION);
	executed_flops = static_cast<double>(elements) * iterations *
		OP_COUNT * FLOPS_PER_ELEMENT_OP;
	actual_working_set_mib = static_cast<double>(2ULL * elements * sizeof(PRECISION)) /
		(1024.0 * 1024.0);
	CUDA_CHECK(cudaFree(buffer_a));
	CUDA_CHECK(cudaFree(buffer_b));
#endif

	const float elapsed_ms = median(time_series);
	const double seconds = elapsed_ms / 1e3;
	const double gflops = executed_flops / seconds / 1e9;
	const double bandwidth = transferred_bytes / seconds / 1e9;

	cout << gflops << " GFLOP/s " << bandwidth << " GB/s " << EFFECTIVE_AI
		 << " FLOP/byte " << actual_working_set_mib << " MiB" << endl;

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	return 0;
}
