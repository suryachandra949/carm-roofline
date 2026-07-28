#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

// DEFINE NUM_REPS
// DEFINE KERNEL PARAMETERS
// DEFINE PRECISION
// DEFINE DEVICE
// DEFINE TEST

__global__ void benchmark(PRECISION *sink, int iterations) {
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	volatile __shared__ PRECISION source[THREADS_PER_BLOCK];
	volatile __shared__ PRECISION destination[THREADS_PER_BLOCK];

	// DEFINE INITIALIZATION

	source[threadIdx.x] = fp_bias;
	destination[threadIdx.x] = fp_bias;
	__syncthreads();

	PRECISION memory_value = fp_bias;
	for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
		for (int inner = 0; inner < 128; ++inner) {
			// DEFINE FP FIRST
			memory_value = source[threadIdx.x];
			// DEFINE FP SECOND
			destination[threadIdx.x] = memory_value;
		}
	}

	sink[id] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

static float median(vector<float> values) {
	sort(values.begin(), values.end());
	const size_t middle = values.size() / 2;
	return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0f
	                              : values[middle];
}

int main() {
	cudaSetDevice(DEVICE);
	const uint64_t launched_threads = static_cast<uint64_t>(NUM_BLOCKS) * THREADS_PER_BLOCK;

	PRECISION *sink = nullptr;
	cudaMalloc(reinterpret_cast<void **>(&sink), launched_threads * sizeof(PRECISION));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int iterations = 1;
	float milliseconds = 0.0f;
	while (milliseconds < 150.0f) {
		iterations *= 2;
		cudaEventRecord(start);
		benchmark<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(sink, iterations);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
	}

	vector<float> samples;
	samples.reserve(NUM_REPS);
	for (int repetition = 0; repetition < NUM_REPS; ++repetition) {
		cudaEventRecord(start);
		benchmark<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(sink, iterations);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		samples.push_back(milliseconds);
	}

	const double elapsed_seconds = median(samples) / 1000.0;
	const double work_items = static_cast<double>(iterations) * 128.0 * launched_threads;
	const double total_flops = work_items * NUM_FP * FLOPS_PER_INST * PACKED_LANES;
	const double total_requested_bytes = work_items * 2.0 * sizeof(PRECISION);
	const double arithmetic_intensity = total_flops / total_requested_bytes;
	const double gflops = total_flops / elapsed_seconds / 1.0e9;
	const double gbps = total_requested_bytes / elapsed_seconds / 1.0e9;

	cout << "AI=" << arithmetic_intensity << ",GFLOPS=" << gflops << ",GBPS=" << gbps
	     << ",FLOPS=" << total_flops << ",BYTES=" << total_requested_bytes << endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(sink);
	return 0;
}
