# NVIDIA mixed benchmark

The mixed benchmark executes vector arithmetic while moving data through a selected GPU memory level. Arithmetic intensity (AI) is expressed in FLOPs per byte.

## A40 L2 example

```bash
python3 run_gpu.py --test mixed --vector sp --vector_op fma \
  --mixed_target L2 --ai 32 --threads 256 --blocks 4096 \
  --iterations 200 --warmup_iterations 25 --l2_fraction 0.50 \
  --name A40
```

The L2 path is designed for repeated, cache-resident measurements:

1. It allocates one in-place working set rather than two alternating buffers.
2. The working set is a configurable fraction of detected L2.
3. It is capped by `persistingL2CacheMaxSize` and `accessPolicyMaxWindowSize`.
4. It requests a persisting-L2 access-policy window with hit ratio 1.0.
5. It compiles global loads with `-Xptxas -dlcm=cg`, bypassing L1 and caching at L2.
6. It runs untimed warm-up launches before any timing.
7. It times every requested launch and reports the average and timing variation.

A `--l2_fraction 0.50` setting on a GPU with 6 MiB of L2 produces a working set near 3 MiB. This smaller footprint is deliberate: using the whole cache cannot guarantee residency because of replacement, set conflicts, other contexts, and cache bookkeeping.

The CUDA persistence controls are hints, not an absolute cache-locking guarantee. Verify a near-100% L2 read hit rate with Nsight Compute.

## Output

```text
<GFLOP/s> GFLOP/s <GB/s> GB/s <effective AI> FLOP/byte
<working set> MiB <average> ms/iteration <stddev> ms_stddev
<minimum> ms_min <maximum> ms_max <iterations> iterations <warmups> warmup
```

Results are appended to `Results/Mixed/<name>_Mixed.csv`.

## Targets

- `global`: two-buffer streaming through the requested total working set.
- `L2`: one-buffer in-place accesses with L1-bypassing loads and persisting-L2 hints.
- `shared`: volatile shared-memory accesses; currently supports `sp` and `dp`.

The source-level AI assumes one load and one store per element. Use Nsight Compute counters for hardware-measured L2 AI and L2 hit rate.
