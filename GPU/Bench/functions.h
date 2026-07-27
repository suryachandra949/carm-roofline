#include <cstdint>
#include <string>
using namespace std;
void create_benchmark_flops(int device, string arch, string compute_capability, string operation,
							string precision, int threads_per_block, int num_blocks);

void create_benchmark_tensor(int device, string compute_capability, string precision,
							 int threads_per_block, int num_blocks);

void create_benchmark_matrix(int device, string compute_capability, string precision,
							 int threads_per_block, int num_blocks);

void create_benchmark_mem(int device, string arch, string compute_capability, string target,
						  string precision, int threads_per_block, int num_blocks);
void create_benchmark_mixed(int device, string arch, string compute_capability, string target,
							string operation, string precision, double arithmetic_intensity,
							uint64_t working_set_mb, int threads_per_block, int num_blocks,
							int measured_iterations, int warmup_iterations, double l2_fraction);
