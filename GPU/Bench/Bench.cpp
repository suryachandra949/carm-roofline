#include <getopt.h>
#include <stdlib.h>

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string>

#include "functions.h"

#define no_argument 0
#define required_argument 1
#define optional_argument 2

using namespace std;

int main(int argc, char* argv[]) {
	const struct option longopts[] = {{"test", required_argument, 0, 't'},
									  {"compute", required_argument, 0, 'c'},
									  {"target", required_argument, 0, 'i'},
									  {"arch", required_argument, 0, 'a'},
									  {"precision", required_argument, 0, 'p'},
									  {"operation", required_argument, 0, 'o'},
									  {"help", no_argument, 0, 'h'},
									  {"threads", required_argument, 0, 's'},
									  {"blocks", required_argument, 0, 'b'},
									  {"device", required_argument, 0, 'd'},
									  {"ai", required_argument, 0, 'r'},
									  {"working-set-mb", required_argument, 0, 'w'},
									  {"iterations", required_argument, 0, 'n'},
									  {"warmup-iterations", required_argument, 0, 'u'},
									  {"l2-fraction", required_argument, 0, 'f'},
									  {0, 0, 0, 0}};

	int o;

	string test, target, precision, operation, arch, compute_capability;
	int threads_per_block = 0, num_blocks = 0;
	int DEVICE = 0;
	double arithmetic_intensity = 0.0;
	uint64_t working_set_mb = 0;
	int measured_iterations = 100;
	int warmup_iterations = 20;
	double l2_fraction = 0.50;

	while ((o = getopt_long(argc, argv, "t:c:i:a:p:o:hs:b:d:r:w:n:u:f:", longopts, NULL)) != -1) switch (o) {
			case 't':
				test = optarg;
				break;
			case 'c':
				compute_capability = optarg;
				break;
			case 'i':
				target = optarg;
				break;
			case 'a':
				arch = optarg;
				break;
			case 'p':
				precision = optarg;
				break;
			case 'o':
				operation = optarg;	 // fma, mul, add, div for cuda core operations
				break;
			case 's':
				threads_per_block = atoi(optarg);
				break;
			case 'b':
				num_blocks = atoi(optarg);
				break;
			case 'd':
				DEVICE = atoi(optarg);
				break;
			case 'r':
				arithmetic_intensity = strtod(optarg, NULL);
				break;
			case 'w':
				working_set_mb = strtoull(optarg, NULL, 10);
				break;
			case 'n':
				measured_iterations = atoi(optarg);
				break;
			case 'u':
				warmup_iterations = atoi(optarg);
				break;
			case 'f':
				l2_fraction = strtod(optarg, NULL);
				break;
			case 'h':
				// TODO: IMPLEMENT
				// fprintf(stdout,
				//         "Usage: %s [OPTION]...\n\n\t-m \t M dimension [int] "
				//         "[default=1024]\n\t-n \t N "
				//         "dimension [int] [default=1024]\n\t-k \t K dimension [int] "
				//         "[default=1024]\n\t-a \t All "
				//         "dimensions [int]\n\t-c \t Disable Tensor Cores\n\n",
				//         argv[0]);
				exit(EXIT_SUCCESS);
			default:
				// TODO:IMPLEMENT
				// fprintf(stderr,
				//         "Usage: %s [OPTION]...\n\n\t-m \t M dimension [int] "
				//         "[default=1024]\n\t-n \t N "
				//         "dimension [int] [default=1024]\n\t-k \t K dimension [int] "
				//         "[default=1024]\n\t-a \t All "
				//         "dimensions [int]\n\t-c \t Disable Tensor Cores\n\n",
				//         argv[0]);
				exit(EXIT_FAILURE);
		}

	if (compute_capability.empty()) {
		cerr << "ERROR: Compute Capability not set. Unable to compile benchmarks." << endl;
		return 3;
	}

	if (test == "FLOPS") {
		if (target == "vector")
			create_benchmark_flops(DEVICE, arch, compute_capability, operation, precision,
								   threads_per_block, num_blocks);
		else if (target == "tensor") {
			if (arch == "nvidia") {
				create_benchmark_tensor(DEVICE, compute_capability, precision, threads_per_block,
									num_blocks);
			} else {
				create_benchmark_matrix(DEVICE, compute_capability, precision, threads_per_block, num_blocks);
			}
		}
	} else if (test == "MEM") {
		create_benchmark_mem(DEVICE, arch, compute_capability, target, precision, threads_per_block,
							 num_blocks);
	} else if (test == "MIXED") {
		create_benchmark_mixed(DEVICE, arch, compute_capability, target, operation, precision,
						   arithmetic_intensity, working_set_mb, threads_per_block, num_blocks,
						   measured_iterations, warmup_iterations, l2_fraction);
	} else {
		cerr << "ERROR: Test not found. Please select a valid test." << endl;
		return 2;
	};

	return 0;
}