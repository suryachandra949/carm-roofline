#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace std;

// DEFINE NUM_REPS
// DEFINE KERNEL PARAMETERS
// DEFINE PRECISION
// DEFINE DEVICE
// DEFINE TARGET
// DEFINE TEST

__global__ void benchmark(const PRECISION *__restrict__ source,
                          PRECISION *__restrict__ destination,
                          PRECISION *__restrict__ sink,
                          uint64_t element_count, int iterations) {
	const uint64_t id = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
	const uint64_t grid_size = gridDim.x * (uint64_t)blockDim.x;

	// DEFINE INITIALIZATION

	for (int iteration = 0; iteration < iterations; ++iteration) {
		for (uint64_t index = id; index < element_count; index += grid_size) {
			// DEFINE FP FIRST
			PRECISION memory_value = source[index];
			// DEFINE FP SECOND
			destination[index] = memory_value;
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
	hipSetDevice(DEVICE);

	hipDeviceProp_t properties;
	hipGetDeviceProperties(&properties, DEVICE);

	size_t free_bytes = 0;
	size_t total_bytes = 0;
	hipMemGetInfo(&free_bytes, &total_bytes);

	const size_t l2_bytes = static_cast<size_t>(properties.l2CacheSize);
	size_t bytes_per_array = 0;
	if (TARGET_L2) {
		bytes_per_array = max<size_t>(sizeof(PRECISION), l2_bytes / 4);
	} else {
		const size_t desired = max<size_t>(4 * l2_bytes, 256ULL << 20);
		bytes_per_array = min(desired, free_bytes / 4);
	}

	const uint64_t element_count = bytes_per_array / sizeof(PRECISION);
	if (element_count == 0) {
		cerr << "ERROR: Mixed benchmark working set is empty." << endl;
		return 1;
	}

	const int launch_blocks = TARGET_L2 ? max(1, 2 * properties.multiProcessorCount) : NUM_BLOCKS;
	const int launch_threads = THREADS_PER_BLOCK;
	const uint64_t launched_threads = static_cast<uint64_t>(launch_blocks) * launch_threads;

	if (!TARGET_L2 && 2 * bytes_per_array <= 2 * l2_bytes) {
		cerr << "WARNING: Global mixed working set may not exceed L2." << endl;
	}

	PRECISION *source = nullptr;
	PRECISION *destination = nullptr;
	PRECISION *sink = nullptr;
	hipMalloc(reinterpret_cast<void **>(&source), element_count * sizeof(PRECISION));
	hipMalloc(reinterpret_cast<void **>(&destination), element_count * sizeof(PRECISION));
	hipMalloc(reinterpret_cast<void **>(&sink), launched_threads * sizeof(PRECISION));
	hipMemset(source, 0, element_count * sizeof(PRECISION));
	hipMemset(destination, 0, element_count * sizeof(PRECISION));

	hipEvent_t start, stop;
	hipEventCreate(&start);
	hipEventCreate(&stop);

	int iterations = 1;
	float milliseconds = 0.0f;
	while (milliseconds < 150.0f) {
		iterations *= 2;
		hipEventRecord(start);
		benchmark<<<launch_blocks, launch_threads>>>(source, destination, sink, element_count,
		                                           iterations);
		hipEventRecord(stop);
		hipEventSynchronize(stop);
		hipEventElapsedTime(&milliseconds, start, stop);
	}

	vector<float> samples;
	samples.reserve(NUM_REPS);
	for (int repetition = 0; repetition < NUM_REPS; ++repetition) {
		hipEventRecord(start);
		benchmark<<<launch_blocks, launch_threads>>>(source, destination, sink, element_count,
		                                           iterations);
		hipEventRecord(stop);
		hipEventSynchronize(stop);
		hipEventElapsedTime(&milliseconds, start, stop);
		samples.push_back(milliseconds);
	}

	const double elapsed_seconds = median(samples) / 1000.0;
	const double work_items = static_cast<double>(iterations) * element_count;
	const double total_flops = work_items * NUM_FP * FLOPS_PER_INST * PACKED_LANES;
	const double total_requested_bytes = work_items * 2.0 * sizeof(PRECISION);
	const double arithmetic_intensity = total_flops / total_requested_bytes;
	const double gflops = total_flops / elapsed_seconds / 1.0e9;
	const double gbps = total_requested_bytes / elapsed_seconds / 1.0e9;

	cout << "AI=" << arithmetic_intensity << ",GFLOPS=" << gflops << ",GBPS=" << gbps
	     << ",FLOPS=" << total_flops << ",BYTES=" << total_requested_bytes << endl;

	hipEventDestroy(start);
	hipEventDestroy(stop);
	hipFree(source);
	hipFree(destination);
	hipFree(sink);
	return 0;
}
