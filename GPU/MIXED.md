# NVIDIA mixed benchmark

The mixed benchmark executes vector arithmetic while moving data through a selected GPU memory level. It is intended to generate points at a requested arithmetic intensity (AI), expressed in FLOPs per byte.

## Example: NVIDIA A40, FP32 global-memory AI of 32 FLOP/byte

```bash
python3 run_gpu.py --test mixed --vector sp --vector_op fma \
  --mixed_target global --ai 32 --working_set_mb 512 \
  --threads 256 --blocks 4096 --name A40
```

The executable reports:

```text
<GFLOP/s> GFLOP/s <GB/s> GB/s <effective AI> FLOP/byte <working set> MiB
```

Results are also appended to `Results/Mixed/<name>_Mixed.csv`.

## Targets

- `global`: streams through the requested total two-buffer working set. Use a working set much larger than L2 when you want a DRAM roofline point.
- `L2`: automatically sizes the two buffers to half of the detected L2 cache.
- `shared`: measures arithmetic mixed with volatile shared-memory loads and stores. This path supports `sp` and `dp`.

## Precision and AI quantization

The global and L2 paths support `sp`, `dp`, `hp`, `hp2`, and `bf16`. Integer arithmetic is intentionally excluded because the requested unit is FLOPs/byte.

The operation count is integral, so the achieved AI may differ slightly from the requested value. The tool reports both values. For FP32 FMA, the AI increment is 0.25 FLOP/byte because one FMA is two FLOPs and each element causes one 4-byte load plus one 4-byte store.

The source-level AI assumes one load and one store per element. For a hardware-level DRAM or L2 AI, validate traffic and executed instructions with Nsight Compute counters.
