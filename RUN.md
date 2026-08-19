# NeuroCompiler Run Guide

This project trains and evaluates a hybrid LLVM optimizer.

The hybrid model has two parts:

- SL model: scores which LLVM pass looks useful for the current program state.
- RL model: chooses the pass sequence to apply.

## Setup

Run everything from WSL:

```bash
cd "/mnt/c/Users/nithi/OneDrive/Documents/ChatGPT/Neurocompiler/Final-Year-Project-main"
conda activate neurocompiler
```

## Train The RL Model

This trains the RL pass-sequence model using runtime reward data:

```bash
python training/train_rl.py \
  --input datasets/replay_buffer/rl_experiences_runtime_full_z.csv \
  --output-dir models/reinforcement_runtime_full_new \
  --gamma 0.90 \
  --q-iterations 3
```

Input data:

```text
datasets/replay_buffer/rl_experiences_runtime_full_z.csv
```

Output model:

```text
models/reinforcement_runtime_full_new
```

## Test Runtime Against Clang O3

This measures the trained hybrid model against the `clang -O3` baseline:

```bash
python evaluation/o3_runtime_harness.py measure \
  --benchmarks benchmark://cbench-v1/bitcount \
  --benchmarks benchmark://cbench-v1/bzip2 \
  --benchmarks benchmark://cbench-v1/dijkstra \
  --benchmarks benchmark://cbench-v1/gsm \
  --benchmarks benchmark://cbench-v1/ispell \
  --benchmarks benchmark://cbench-v1/jpeg-c \
  --benchmarks benchmark://cbench-v1/lame \
  --benchmarks benchmark://cbench-v1/stringsearch \
  --benchmarks benchmark://cbench-v1/tiff2bw \
  --benchmarks benchmark://cbench-v1/tiff2rgba \
  --sl-model-dir models/supervised_runtime \
  --rl-model-dir models/reinforcement_runtime_full_new \
  --max-steps 15 \
  --warmup 1 \
  --reps 3 \
  --cpu 4 \
  --timeout 180 \
  --inputs 0 \
  --workdir results/o3_harness_work_runtime_full_new \
  --output results/o3_runtime_full_new_live10.json \
  --include-fallback
```

## Summarize Results

```bash
python evaluation/o3_runtime_harness.py summarize \
  --results results/o3_runtime_full_new_live10.json \
  --output results/o3_runtime_full_new_live10_summary.json
```

Important output line:

```text
Geo-mean speedup hybrid vs clang -O3
```

If the value is above `1.0000x`, the hybrid model is faster overall.

## Latest Local Result

Newly trained RL model:

```text
Hybrid vs clang -O3: 1.0145x
Approx improvement: +1.45%
Wins: 7 / 10
```

Older saved runtime model:

```text
Hybrid vs clang -O3: 1.0235x
Approx improvement: +2.35%
Wins: 7 / 10
```

Current next goal:

```text
Improve the data/reward/generalization so hybrid reduces the remaining benchmark regressions.
```
