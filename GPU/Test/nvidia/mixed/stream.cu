#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>

using namespace std;

// Generator placeholders. Keep these lines unchanged.
// DEFINE KERNEL PARAMETERS
// DEFINE PRECISION
// DEFINE DEVICE
// DEFINE TARGET
// DEFINE TEST

/*
 * Fixed measurement settings.
 *
 * Increase MIXED_MEASUREMENT_REPS if the reported SECONDS value is too short.
 */
#ifndef MIXED_KERNEL_ITERATIONS
#define MIXED_KERNEL_ITERATIONS 100000
#endif

#ifndef MIXED_WARMUP_REPS
#define MIXED_WARMUP_REPS 2
#endif

#ifndef MIXED_MEASUREMENT_REPS
#define MIXED_MEASUREMENT_REPS 256
#endif

#define CUDA_CHECK(call)                                                      \
	do {                                                                      \
		const cudaError_t error_code = (call);                                 \
		if (error_code != cudaSuccess) {                                       \
			cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "    \
			     << cudaGetErrorString(error_code) << endl;                    \
			return 1;                                                          \
		}                                                                      \
	} while (0)

__global__ void benchmark(
	const PRECISION *__restrict__ source,
	PRECISION *__restrict__ destination,
	PRECISION *__restrict__ sink,
	uint64_t element_count,
	int iterations
) {
	const uint64_t id =
		blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;

	const uint64_t grid_size =
		gridDim.x * static_cast<uint64_t>(blockDim.x);

	// DEFINE INITIALIZATION

	for (int iteration = 0; iteration < iterations; ++iteration) {
		for (uint64_t index = id;
		     index < element_count;
		     index += grid_size) {

			// DEFINE FP FIRST

			const PRECISION memory_value = source[index];

			// DEFINE FP SECOND

			destination[index] = memory_value;
		}
	}

	/*
	 * Keep the eight independent FP chains live.
	 *
	 * This sink operation has a small overhead that is intentionally excluded
	 * from the main-loop FLOP and byte accounting.
	 */
	sink[id] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

int main() {
	CUDA_CHECK(cudaSetDevice(DEVICE));

	cudaDeviceProp properties{};
	CUDA_CHECK(cudaGetDeviceProperties(&properties, DEVICE));

	size_t free_bytes = 0;
	size_t total_bytes = 0;
	CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

	const size_t l2_bytes =
		static_cast<size_t>(properties.l2CacheSize);

	size_t bytes_per_array = 0;

	if (TARGET_L2) {
		/*
		 * The source and destination arrays together occupy approximately
		 * half of the L2 cache.
		 */
		bytes_per_array = max<size_t>(
			sizeof(PRECISION),
			l2_bytes / 4
		);
	} else {
		/*
		 * Use a working set substantially larger than L2 for the global
		 * memory test.
		 */
		const size_t desired_bytes_per_array = max<size_t>(
			4 * l2_bytes,
			256ULL << 20
		);

		/*
		 * Each array may use at most one quarter of currently free memory.
		 * This leaves space for the second array, sink array and runtime
		 * allocations.
		 */
		bytes_per_array = min(
			desired_bytes_per_array,
			free_bytes / 4
		);
	}

	const uint64_t element_count =
		bytes_per_array / sizeof(PRECISION);

	if (element_count == 0) {
		cerr << "ERROR: Mixed benchmark working set is empty." << endl;
		return 1;
	}

	const int launch_blocks = TARGET_L2
		? max(1, 2 * properties.multiProcessorCount)
		: NUM_BLOCKS;

	const int launch_threads = THREADS_PER_BLOCK;

	const uint64_t launched_threads =
		static_cast<uint64_t>(launch_blocks) *
		static_cast<uint64_t>(launch_threads);

	if (!TARGET_L2 &&
	    2 * bytes_per_array <= 2 * l2_bytes) {
		cerr << "WARNING: Global mixed working set may not exceed L2."
		     << endl;
	}

	PRECISION *source = nullptr;
	PRECISION *destination = nullptr;
	PRECISION *sink = nullptr;

	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void **>(&source),
		element_count * sizeof(PRECISION)
	));

	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void **>(&destination),
		element_count * sizeof(PRECISION)
	));

	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void **>(&sink),
		launched_threads * sizeof(PRECISION)
	));

	CUDA_CHECK(cudaMemset(
		source,
		0,
		element_count * sizeof(PRECISION)
	));

	CUDA_CHECK(cudaMemset(
		destination,
		0,
		element_count * sizeof(PRECISION)
	));

	CUDA_CHECK(cudaMemset(
		sink,
		0,
		launched_threads * sizeof(PRECISION)
	));

	cudaEvent_t start = nullptr;
	cudaEvent_t stop = nullptr;

	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	const int iterations = MIXED_KERNEL_ITERATIONS;
	const int warmup_reps = MIXED_WARMUP_REPS;
	const int measurement_reps = MIXED_MEASUREMENT_REPS;

	/*
	 * Warm-up phase.
	 *
	 * Nothing in this phase is included in the reported time, FLOPs or bytes.
	 */
	for (int repetition = 0;
	     repetition < warmup_reps;
	     ++repetition) {

		benchmark<<<launch_blocks, launch_threads>>>(
			source,
			destination,
			sink,
			element_count,
			iterations
		);
	}

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	/*
	 * The ENERGY_BEGIN and ENERGY_END messages are markers only.
	 * Replace them or connect them to the API used by the energy-measurement
	 * system.
	 */
	cerr << "ENERGY_BEGIN"
	     << ",ITERATIONS=" << iterations
	     << ",KERNELS=" << measurement_reps
	     << endl;

	CUDA_CHECK(cudaEventRecord(start));

	for (int repetition = 0;
	     repetition < measurement_reps;
	     ++repetition) {

		benchmark<<<launch_blocks, launch_threads>>>(
			source,
			destination,
			sink,
			element_count,
			iterations
		);
	}

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	cerr << "ENERGY_END" << endl;

	float total_milliseconds = 0.0f;

	CUDA_CHECK(cudaEventElapsedTime(
		&total_milliseconds,
		start,
		stop
	));

	const double elapsed_seconds =
		static_cast<double>(total_milliseconds) / 1000.0;

	if (elapsed_seconds <= 0.0) {
		cerr << "ERROR: Invalid measured execution time." << endl;

		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		cudaFree(source);
		cudaFree(destination);
		cudaFree(sink);

		return 1;
	}

	/*
	 * Across all threads, one kernel iteration collectively processes
	 * element_count elements.
	 */
	const double work_items =
		static_cast<double>(measurement_reps) *
		static_cast<double>(iterations) *
		static_cast<double>(element_count);

	/*
	 * NUM_FP is the number of generated FP instructions per work item.
	 *
	 * FLOPS_PER_INST:
	 *   FMA     -> 2
	 *   ADD/MUL -> 1
	 *
	 * PACKED_LANES:
	 *   half2 -> 2
	 *   other -> 1
	 */
	const double total_flops =
		work_items *
		static_cast<double>(NUM_FP) *
		static_cast<double>(FLOPS_PER_INST) *
		static_cast<double>(PACKED_LANES);

	/*
	 * Every work item performs one source load and one destination store.
	 */
	const double total_requested_bytes =
		work_items *
		2.0 *
		static_cast<double>(sizeof(PRECISION));

	const double arithmetic_intensity =
		total_flops / total_requested_bytes;

	const double gflops =
		total_flops / elapsed_seconds / 1.0e9;

	const double gbps =
		total_requested_bytes / elapsed_seconds / 1.0e9;

	if (elapsed_seconds < 1.0) {
		cerr
			<< "WARNING: Measurement lasted only "
			<< elapsed_seconds
			<< " seconds. Increase MIXED_MEASUREMENT_REPS for "
			   "more stable energy measurements."
			<< endl;
	}

	cout << "AI=" << arithmetic_intensity
	     << ",GFLOPS=" << gflops
	     << ",GBPS=" << gbps
	     << ",FLOPS=" << total_flops
	     << ",BYTES=" << total_requested_bytes
	     << ",SECONDS=" << elapsed_seconds
	     << ",ITERATIONS=" << iterations
	     << ",KERNELS=" << measurement_reps
	     << endl;

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	CUDA_CHECK(cudaFree(source));
	CUDA_CHECK(cudaFree(destination));
	CUDA_CHECK(cudaFree(sink));

	return 0;
}