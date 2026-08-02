#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

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
 * Warning: 100000 iterations and 256 measurement launches can produce a very
 * long run. Reduce these values if the measured interval is excessive or the
 * platform has a kernel watchdog.
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

struct HostTimestamp {
	int64_t unix_ns;
	int64_t steady_ns;
	string utc;
};

static HostTimestamp get_host_timestamp() {
	using namespace chrono;

	const system_clock::time_point realtime = system_clock::now();
	const steady_clock::time_point monotonic = steady_clock::now();

	const int64_t unix_ns =
		duration_cast<nanoseconds>(realtime.time_since_epoch()).count();

	const int64_t steady_ns =
		duration_cast<nanoseconds>(monotonic.time_since_epoch()).count();

	const time_t seconds = system_clock::to_time_t(realtime);
	tm utc_time{};
	gmtime_r(&seconds, &utc_time);

	const int64_t fractional_ns =
		((unix_ns % 1000000000LL) + 1000000000LL) % 1000000000LL;

	ostringstream formatted;
	formatted << put_time(&utc_time, "%Y-%m-%dT%H:%M:%S")
	          << "."
	          << setfill('0')
	          << setw(9)
	          << fractional_ns
	          << "Z";

	return {unix_ns, steady_ns, formatted.str()};
}

static uint64_t gcd_u64(uint64_t a, uint64_t b) {
	while (b != 0) {
		const uint64_t remainder = a % b;
		a = b;
		b = remainder;
	}
	return a;
}

static uint64_t lcm_u64(uint64_t a, uint64_t b) {
	if (a == 0 || b == 0) {
		return 0;
	}
	return (a / gcd_u64(a, b)) * b;
}

__global__ void benchmark(
	const PRECISION *__restrict__ source,
	PRECISION *__restrict__ destination,
	uint64_t element_count,
	uint64_t elements_per_block,
	int iterations
) {
	const uint64_t id =
		blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;

	const uint64_t grid_size =
		gridDim.x * static_cast<uint64_t>(blockDim.x);

	// DEFINE INITIALIZATION

	if (TARGET_L2) {
		/*
		 * Every block owns one disjoint source slice and one disjoint
		 * destination slice. L2 is still physically shared by the GPU.
		 */
		const uint64_t block_base =
			static_cast<uint64_t>(blockIdx.x) * elements_per_block;

		for (int iteration = 0; iteration < iterations; ++iteration) {
			for (uint64_t local_index = threadIdx.x;
			     local_index < elements_per_block;
			     local_index += blockDim.x) {

				const uint64_t index = block_base + local_index;

				// DEFINE FP FIRST

				const PRECISION memory_value = source[index];

				// DEFINE FP SECOND

				destination[index] = memory_value;
			}
		}

		/*
		 * Keep every thread's FP chains live without allocating a separate
		 * sink array. elements_per_block is validated to be at least the
		 * number of threads in the block.
		 *
		 * This final store and its reduction additions are excluded from the
		 * reported main-loop FLOP and byte accounting.
		 */
		destination[block_base + threadIdx.x] =
			a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
	} else {
		/*
		 * Global-memory mode keeps the original grid-stride traversal.
		 * All blocks collectively process the full streaming working set.
		 */
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

		if (id < element_count) {
			destination[id] =
				a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
		}
	}
}

int main() {
	CUDA_CHECK(cudaSetDevice(DEVICE));

	cudaDeviceProp properties{};
	CUDA_CHECK(cudaGetDeviceProperties(&properties, DEVICE));

	if (NUM_BLOCKS <= 0) {
		cerr << "ERROR: NUM_BLOCKS must be greater than zero." << endl;
		return 1;
	}

	if (THREADS_PER_BLOCK <= 0 ||
	    THREADS_PER_BLOCK > properties.maxThreadsPerBlock) {
		cerr << "ERROR: Invalid THREADS_PER_BLOCK="
		     << THREADS_PER_BLOCK
		     << "; device maximum is "
		     << properties.maxThreadsPerBlock
		     << "."
		     << endl;
		return 1;
	}

	size_t free_bytes = 0;
	size_t total_bytes = 0;
	CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

	const size_t l2_bytes =
		static_cast<size_t>(properties.l2CacheSize);

	const int launch_blocks = NUM_BLOCKS;
	const int launch_threads = THREADS_PER_BLOCK;

	uint64_t element_count = 0;
	uint64_t elements_per_block = 0;
	size_t bytes_per_array = 0;

	if (TARGET_L2) {
		/*
		 * source + destination together use approximately half of L2.
		 *
		 * The total budget is fixed. It is divided equally among the
		 * requested number of blocks, and each block gets one source slice
		 * and one destination slice.
		 */
		const size_t combined_l2_budget =
			max<size_t>(2 * sizeof(PRECISION), l2_bytes / 2);

		const size_t per_array_budget =
			combined_l2_budget / 2;

		const uint64_t raw_elements_per_block =
			static_cast<uint64_t>(
				per_array_budget /
				static_cast<size_t>(launch_blocks) /
				sizeof(PRECISION)
			);

		/*
		 * Make each region:
		 *   1. large enough for every thread to write one final sink value;
		 *   2. divisible evenly among threads;
		 *   3. aligned to a 128-byte region boundary.
		 */
		const uint64_t alignment_elements =
			max<uint64_t>(
				1,
				128ULL / static_cast<uint64_t>(sizeof(PRECISION))
			);

		const uint64_t region_granularity =
			lcm_u64(
				static_cast<uint64_t>(launch_threads),
				alignment_elements
			);

		if (region_granularity == 0) {
			cerr << "ERROR: Invalid L2 region granularity." << endl;
			return 1;
		}

		elements_per_block =
			(raw_elements_per_block / region_granularity) *
			region_granularity;

		if (elements_per_block <
		    static_cast<uint64_t>(launch_threads)) {
			cerr << "ERROR: Too many blocks for the selected L2 budget. "
			     << "Each block would receive only "
			     << raw_elements_per_block
			     << " elements, but "
			     << launch_threads
			     << " threads were requested."
			     << endl;
			return 1;
		}

		element_count =
			static_cast<uint64_t>(launch_blocks) *
			elements_per_block;

		bytes_per_array =
			static_cast<size_t>(element_count) *
			sizeof(PRECISION);
	} else {
		/*
		 * Use a working set substantially larger than L2 for global memory.
		 */
		const size_t desired_bytes_per_array =
			max<size_t>(
				4 * l2_bytes,
				256ULL << 20
			);

		/*
		 * Each array may use at most one quarter of currently free memory.
		 * This leaves room for the second array and CUDA runtime activity.
		 */
		bytes_per_array =
			min(
				desired_bytes_per_array,
				free_bytes / 4
			);

		element_count =
			bytes_per_array / sizeof(PRECISION);

		if (element_count == 0) {
			cerr << "ERROR: Global mixed working set is empty." << endl;
			return 1;
		}

		if (2 * bytes_per_array <= 2 * l2_bytes) {
			cerr << "WARNING: Global mixed working set may not exceed L2."
			     << endl;
		}
	}

	if (element_count == 0 || bytes_per_array == 0) {
		cerr << "ERROR: Mixed benchmark working set is empty." << endl;
		return 1;
	}

	PRECISION *source = nullptr;
	PRECISION *destination = nullptr;

	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void **>(&source),
		bytes_per_array
	));

	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void **>(&destination),
		bytes_per_array
	));

	CUDA_CHECK(cudaMemset(
		source,
		0,
		bytes_per_array
	));

	CUDA_CHECK(cudaMemset(
		destination,
		0,
		bytes_per_array
	));

	cudaEvent_t start = nullptr;
	cudaEvent_t stop = nullptr;

	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	const int iterations = MIXED_KERNEL_ITERATIONS;
	const int warmup_reps = MIXED_WARMUP_REPS;
	const int measurement_reps = MIXED_MEASUREMENT_REPS;

	cout << "CONFIG"
	     << ",TARGET=" << (TARGET_L2 ? "L2" : "GLOBAL")
	     << ",SM_COUNT=" << properties.multiProcessorCount
	     << ",BLOCKS=" << launch_blocks
	     << ",THREADS_PER_BLOCK=" << launch_threads
	     << ",ELEMENTS_PER_BLOCK="
	     << (TARGET_L2 ? elements_per_block : 0)
	     << ",SOURCE_BYTES_PER_BLOCK="
	     << (TARGET_L2
	             ? elements_per_block * sizeof(PRECISION)
	             : 0)
	     << ",DESTINATION_BYTES_PER_BLOCK="
	     << (TARGET_L2
	             ? elements_per_block * sizeof(PRECISION)
	             : 0)
	     << ",TOTAL_ELEMENTS=" << element_count
	     << ",BYTES_PER_ARRAY=" << bytes_per_array
	     << ",TOTAL_WORKING_SET_BYTES=" << 2 * bytes_per_array
	     << ",L2_BYTES=" << l2_bytes
	     << endl;

	/*
	 * Warm-up phase. It is excluded from the reported time, FLOPs and bytes.
	 */
	for (int repetition = 0;
	     repetition < warmup_reps;
	     ++repetition) {

		benchmark<<<launch_blocks, launch_threads>>>(
			source,
			destination,
			element_count,
			elements_per_block,
			iterations
		);
	}

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	/*
	 * Print and flush the host timestamp before recording the CUDA start
	 * event. This makes the host window slightly wider than the GPU-event
	 * interval, which is useful when correlating external energy samples.
	 */
	const HostTimestamp host_start = get_host_timestamp();

	cout << "ENERGY_START"
	     << ",UNIX_NS=" << host_start.unix_ns
	     << ",STEADY_NS=" << host_start.steady_ns
	     << ",UTC=" << host_start.utc
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
			element_count,
			elements_per_block,
			iterations
		);
	}

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	const HostTimestamp host_stop = get_host_timestamp();

	cout << "ENERGY_STOP"
	     << ",UNIX_NS=" << host_stop.unix_ns
	     << ",STEADY_NS=" << host_stop.steady_ns
	     << ",UTC=" << host_stop.utc
	     << endl;

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

		return 1;
	}

	const int64_t host_window_ns =
		host_stop.unix_ns - host_start.unix_ns;

	const int64_t steady_window_ns =
		host_stop.steady_ns - host_start.steady_ns;

	/*
	 * In L2 mode:
	 *   work per kernel iteration =
	 *       launch_blocks * elements_per_block
	 *
	 * In global mode:
	 *   work per kernel iteration =
	 *       element_count
	 *
	 * Both are represented by element_count after host-side setup.
	 */
	const double work_items =
		static_cast<double>(measurement_reps) *
		static_cast<double>(iterations) *
		static_cast<double>(element_count);

	const double total_flops =
		work_items *
		static_cast<double>(NUM_FP) *
		static_cast<double>(FLOPS_PER_INST) *
		static_cast<double>(PACKED_LANES);

	/*
	 * Every main-loop work item performs one source load and one destination
	 * store. The final per-thread sink stores are intentionally excluded.
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
		cerr << "WARNING: Measurement lasted only "
		     << elapsed_seconds
		     << " seconds. Increase MIXED_MEASUREMENT_REPS for more "
		        "stable energy measurements."
		     << endl;
	}

	cout << "TIMING"
	     << ",HOST_WINDOW_NS=" << host_window_ns
	     << ",STEADY_WINDOW_NS=" << steady_window_ns
	     << ",GPU_ELAPSED_MS=" << total_milliseconds
	     << endl;

	/*
	 * Keep this as the final stdout line: run_gpu.py parses the final
	 * non-empty line as the mixed benchmark result.
	 */
	cout << "AI=" << arithmetic_intensity
	     << ",GFLOPS=" << gflops
	     << ",GBPS=" << gbps
	     << ",FLOPS=" << total_flops
	     << ",BYTES=" << total_requested_bytes
	     << ",SECONDS=" << elapsed_seconds
	     << ",ITERATIONS=" << iterations
	     << ",KERNELS=" << measurement_reps
	     << ",BLOCKS=" << launch_blocks
	     << ",THREADS=" << launch_threads
	     << ",ELEMENTS_PER_BLOCK="
	     << (TARGET_L2 ? elements_per_block : 0)
	     << endl;

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	CUDA_CHECK(cudaFree(source));
	CUDA_CHECK(cudaFree(destination));
}