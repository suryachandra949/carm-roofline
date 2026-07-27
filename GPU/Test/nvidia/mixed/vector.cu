#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

// DEFINE KERNEL PARAMETERS
#define VALUES_PER_THREAD 4

// DEFINE PRECISION

// DEFINE DEVICE

// DEFINE MIXED PARAMETERS
#define MEMORY_TARGET_GLOBAL 0
#define MEMORY_TARGET_L2 1
#define MEMORY_TARGET_SHARED 2

// DEFINE BENCHMARK CONTROL

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

#if MEMORY_TARGET == MEMORY_TARGET_GLOBAL
__global__ void benchmark_global(const PRECISION *input, PRECISION *output, uint64_t elements) {
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
#elif MEMORY_TARGET == MEMORY_TARGET_L2
__global__ void benchmark_l2(PRECISION *data, uint64_t elements) {
	const uint64_t thread_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;

	for (uint64_t base = thread_id; base < elements; base += stride * VALUES_PER_THREAD) {
		PRECISION value0 = data[base];
		PRECISION value1 = data[base + stride];
		PRECISION value2 = data[base + 2 * stride];
		PRECISION value3 = data[base + 3 * stride];

#pragma unroll 4
		for (int operation = 0; operation < OP_COUNT; ++operation) {
			value0 = mixed_operation(value0);
			value1 = mixed_operation(value1);
			value2 = mixed_operation(value2);
			value3 = mixed_operation(value3);
		}

		data[base] = value0;
		data[base + stride] = value1;
		data[base + 2 * stride] = value2;
		data[base + 3 * stride] = value3;
	}
}
#else
__global__ void benchmark_shared(PRECISION *sink) {
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
	__syncthreads();

	sink[global_base] = storage[local0];
	sink[global_base + 1] = storage[local1];
	sink[global_base + 2] = storage[local2];
	sink[global_base + 3] = storage[local3];
}
#endif

static double mean(const vector<float>& values) {
	return accumulate(values.begin(), values.end(), 0.0) /
		static_cast<double>(values.size());
}

static double standard_deviation(const vector<float>& values, double average) {
	double sum = 0.0;
	for (float value : values) {
		const double difference = static_cast<double>(value) - average;
		sum += difference * difference;
	}
	return sqrt(sum / static_cast<double>(values.size()));
}

int main() {
	CUDA_CHECK(cudaSetDevice(DEVICE));

	cudaDeviceProp properties;
	CUDA_CHECK(cudaGetDeviceProperties(&properties, DEVICE));

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	vector<cudaEvent_t> events(MEASURED_ITERATIONS + 1);
	for (cudaEvent_t& event : events) CUDA_CHECK(cudaEventCreate(&event));

	uint64_t elements = 0;
	uint64_t launch_blocks = 0;
	double bytes_per_iteration = 0.0;
	double flops_per_iteration = 0.0;
	double actual_working_set_mib = 0.0;
	bool l2_policy_enabled = false;
	PRECISION *allocation = nullptr;
	PRECISION *buffer_a = nullptr;
	PRECISION *buffer_b = nullptr;

#if MEMORY_TARGET == MEMORY_TARGET_SHARED
	launch_blocks = NUM_BLOCKS;
	elements = launch_blocks * THREADS_PER_BLOCK * VALUES_PER_THREAD;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&allocation), elements * sizeof(PRECISION)));
	CUDA_CHECK(cudaMemsetAsync(allocation, 0, elements * sizeof(PRECISION), stream));

	for (int warmup = 0; warmup < WARMUP_ITERATIONS; ++warmup)
		benchmark_shared<<<launch_blocks, THREADS_PER_BLOCK, 0, stream>>>(allocation);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaEventRecord(events[0], stream));
	for (int iteration = 0; iteration < MEASURED_ITERATIONS; ++iteration) {
		benchmark_shared<<<launch_blocks, THREADS_PER_BLOCK, 0, stream>>>(allocation);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaEventRecord(events[iteration + 1], stream));
	}

	bytes_per_iteration = static_cast<double>(elements) * 2.0 * sizeof(PRECISION);
	flops_per_iteration = static_cast<double>(elements) *
		OP_COUNT * FLOPS_PER_ELEMENT_OP;
	actual_working_set_mib =
		static_cast<double>(elements * sizeof(PRECISION)) / (1024.0 * 1024.0);

#elif MEMORY_TARGET == MEMORY_TARGET_L2
	uint64_t requested_bytes =
		static_cast<uint64_t>(static_cast<double>(properties.l2CacheSize) * L2_FRACTION);
	if (properties.persistingL2CacheMaxSize > 0)
		requested_bytes = min<uint64_t>(
			requested_bytes, static_cast<uint64_t>(properties.persistingL2CacheMaxSize));
	if (properties.accessPolicyMaxWindowSize > 0)
		requested_bytes = min<uint64_t>(
			requested_bytes, static_cast<uint64_t>(properties.accessPolicyMaxWindowSize));

	const uint64_t values_per_block = THREADS_PER_BLOCK * VALUES_PER_THREAD;
	uint64_t requested_elements = requested_bytes / sizeof(PRECISION);
	launch_blocks = requested_elements / values_per_block;
	launch_blocks = max<uint64_t>(1, min<uint64_t>(NUM_BLOCKS, launch_blocks));
	const uint64_t group_size = launch_blocks * values_per_block;
	elements = (requested_elements / group_size) * group_size;
	if (elements == 0) elements = group_size;

	const size_t allocation_bytes = elements * sizeof(PRECISION);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&allocation), allocation_bytes));
	CUDA_CHECK(cudaMemsetAsync(allocation, 0, allocation_bytes, stream));

	if (properties.persistingL2CacheMaxSize > 0 &&
		properties.accessPolicyMaxWindowSize > 0) {
		const size_t set_aside = min<size_t>(
			static_cast<size_t>(properties.l2CacheSize),
			static_cast<size_t>(properties.persistingL2CacheMaxSize));
		CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, set_aside));

		cudaStreamAttrValue attribute{};
		attribute.accessPolicyWindow.base_ptr = allocation;
		attribute.accessPolicyWindow.num_bytes = allocation_bytes;
		attribute.accessPolicyWindow.hitRatio = 1.0;
		attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
		attribute.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
		CUDA_CHECK(cudaStreamSetAttribute(
			stream, cudaStreamAttributeAccessPolicyWindow, &attribute));
		l2_policy_enabled = true;
	}

	for (int warmup = 0; warmup < WARMUP_ITERATIONS; ++warmup)
		benchmark_l2<<<launch_blocks, THREADS_PER_BLOCK, 0, stream>>>(allocation, elements);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaEventRecord(events[0], stream));
	for (int iteration = 0; iteration < MEASURED_ITERATIONS; ++iteration) {
		benchmark_l2<<<launch_blocks, THREADS_PER_BLOCK, 0, stream>>>(allocation, elements);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaEventRecord(events[iteration + 1], stream));
	}

	bytes_per_iteration = static_cast<double>(elements) * 2.0 * sizeof(PRECISION);
	flops_per_iteration = static_cast<double>(elements) *
		OP_COUNT * FLOPS_PER_ELEMENT_OP;
	actual_working_set_mib =
		static_cast<double>(elements * sizeof(PRECISION)) / (1024.0 * 1024.0);

#else
	uint64_t requested_elements =
		REQUESTED_WORKING_SET_BYTES / (2ULL * sizeof(PRECISION));
	const uint64_t values_per_block = THREADS_PER_BLOCK * VALUES_PER_THREAD;
	launch_blocks = requested_elements / values_per_block;
	launch_blocks = max<uint64_t>(1, min<uint64_t>(NUM_BLOCKS, launch_blocks));
	const uint64_t group_size = launch_blocks * values_per_block;
	elements = (requested_elements / group_size) * group_size;
	if (elements == 0) elements = group_size;

	const size_t one_buffer_bytes = elements * sizeof(PRECISION);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&allocation), 2ULL * one_buffer_bytes));
	buffer_a = allocation;
	buffer_b = allocation + elements;
	CUDA_CHECK(cudaMemsetAsync(allocation, 0, 2ULL * one_buffer_bytes, stream));

	PRECISION *input = buffer_a;
	PRECISION *output = buffer_b;
	for (int warmup = 0; warmup < WARMUP_ITERATIONS; ++warmup) {
		benchmark_global<<<launch_blocks, THREADS_PER_BLOCK, 0, stream>>>(
			input, output, elements);
		swap(input, output);
	}
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaEventRecord(events[0], stream));
	for (int iteration = 0; iteration < MEASURED_ITERATIONS; ++iteration) {
		benchmark_global<<<launch_blocks, THREADS_PER_BLOCK, 0, stream>>>(
			input, output, elements);
		swap(input, output);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaEventRecord(events[iteration + 1], stream));
	}

	bytes_per_iteration = static_cast<double>(elements) * 2.0 * sizeof(PRECISION);
	flops_per_iteration = static_cast<double>(elements) *
		OP_COUNT * FLOPS_PER_ELEMENT_OP;
	actual_working_set_mib =
		static_cast<double>(2ULL * one_buffer_bytes) / (1024.0 * 1024.0);
#endif

	CUDA_CHECK(cudaEventSynchronize(events.back()));

	vector<float> time_series;
	time_series.reserve(MEASURED_ITERATIONS);
	for (int iteration = 0; iteration < MEASURED_ITERATIONS; ++iteration) {
		float elapsed_ms = 0.0f;
		CUDA_CHECK(cudaEventElapsedTime(
			&elapsed_ms, events[iteration], events[iteration + 1]));
		time_series.push_back(elapsed_ms);
	}

	const double average_ms = mean(time_series);
	const double standard_deviation_ms = standard_deviation(time_series, average_ms);
	const double minimum_ms = *min_element(time_series.begin(), time_series.end());
	const double maximum_ms = *max_element(time_series.begin(), time_series.end());
	const double average_seconds = average_ms / 1e3;
	const double gflops = flops_per_iteration / average_seconds / 1e9;
	const double bandwidth = bytes_per_iteration / average_seconds / 1e9;

	cout << gflops << " GFLOP/s "
		 << bandwidth << " GB/s "
		 << EFFECTIVE_AI << " FLOP/byte "
		 << actual_working_set_mib << " MiB "
		 << average_ms << " ms/iteration "
		 << standard_deviation_ms << " ms_stddev "
		 << minimum_ms << " ms_min "
		 << maximum_ms << " ms_max "
		 << MEASURED_ITERATIONS << " iterations "
		 << WARMUP_ITERATIONS << " warmup" << endl;

	if (l2_policy_enabled) {
		cudaStreamAttrValue attribute{};
		attribute.accessPolicyWindow.num_bytes = 0;
		CUDA_CHECK(cudaStreamSetAttribute(
			stream, cudaStreamAttributeAccessPolicyWindow, &attribute));
		CUDA_CHECK(cudaCtxResetPersistingL2Cache());
	}

	for (cudaEvent_t event : events) CUDA_CHECK(cudaEventDestroy(event));
	CUDA_CHECK(cudaFree(allocation));
	CUDA_CHECK(cudaStreamDestroy(stream));
	return 0;
}
