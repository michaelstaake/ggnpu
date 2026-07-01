# ggnpu

Run GGUF models on AMD NPUs (Krackan / XDNA2).

## Status

| Backend | Status |
|---------|--------|
| AMD NPU | Work in progress |

Validated models (on `npu6` / Krackan): **Llama 3.2 1B** (Q4_K_M), **Qwen2.5-Coder
1.5B** (Q4_K_M), **Gemma 4 E2B** (`gemma4`, Q4_0). Architecture support is
metadata-driven (`general.architecture`), so other `llama`/`qwen2`-family GGUFs
should load; non-pow2 hidden sizes and arbitrary FFN widths are handled by RMSNorm
pad-to-pow2 and SiLU host-tiling. The `gemma4` path adds the full Gemma 4
(MatFormer) feature set — SentencePiece-unigram tokenizer, Per-Layer Embeddings,
QK-norm, sandwich norms, shared-KV, sliding-window attention, and logit soft-cap —
plus a Q4_0 quantization datapath.

## Native host setup

`ggnpu` is intended to be built and run directly on the host. Docker is not used.

### Host prerequisites

| Required on host | Notes |
|------------------|-------|
| Linux with `amdxdna` loaded | Ryzen AI / XDNA-capable host |
| `/dev/accel/accel0` | NPU device node |
| `/usr/lib/firmware/amdnpu` | Firmware directory |
| XRT runtime, headers, tools | `libxrt2`, `libxrt-npu2`, `libxrt-dev`, `libxrt-utils` (xclbinutil) |
| CMake and C++ toolchain | `cmake`, `g++`, `make` or Ninja |
| Python 3.10+ | For kernel compilation via Triton-XDNA |
| User in host `render` group | Required to open the accel device |

BIOS: NPU/IPU enabled. Boot: `amd_iommu=on`.

### Full installation

**One command** — installs host prerequisites, builds ggnpu, optionally builds the
NPU kernels, and verifies. It drives the scripts below, is interactive and safe to
re-run (finished work is skipped):

```bash
git clone https://github.com/michaelstaake/ggnpu.git
cd ggnpu
./setup.sh
```

`./setup.sh` asks before any privileged action. Useful flags: `--yes`
(non-interactive), `--cpu` (CPU-only build, no NPU/driver), `--kernels-here`,
`--kernels-separate`, `--no-kernels`, `--skip-install`, `--skip-build`.
After it runs, **log out and back in (or reboot)** if it added you to the `render`
group or changed memlock limits.

<details>
<summary>Manual alternative (what <code>setup.sh</code> automates)</summary>

```bash
# 1. Core build tools
sudo apt update
sudo apt install build-essential cmake git clang lld ninja-build python3-pip python3-venv

# 2. XRT runtime + NPU driver (from AMD PPA)
sudo add-apt-repository ppa:amd-team/xrt
sudo apt update
sudo apt install libxrt2 libxrt-npu2 libxrt-dev libxrt-utils libxrt-utils-npu amdxdna-dkms

# 3. Memlock limits (required for NPU pinned DMA buffers)
echo '* soft memlock unlimited' | sudo tee /etc/security/limits.d/99-amdxdna.conf
echo '* hard memlock unlimited' | sudo tee -a /etc/security/limits.d/99-amdxdna.conf

# 4. Add user to render group
sudo usermod -aG render $USER

# 5. Log out and back in (or reboot) for group/limits to take effect

# 6. Verify everything
bash scripts/setup-host.sh
bash scripts/verify-npu.sh
```

Steps 1–4 are performed by `scripts/install-host.sh`; the build by `scripts/build.sh`.

</details>

### Quick start

```bash
git clone https://github.com/michaelstaake/ggnpu.git
cd ggnpu

# Build ggnpu with the NPU backend (wrapper; or run the cmake commands below directly)
./scripts/build.sh
# Equivalent:
#   cmake -S . -B build-npu \
#     -DGGNPU_NPU_BACKEND=ON -DGGNPU_TEST_CPU=OFF -DGGNPU_BUILD_TESTS=ON
#   cmake --build build-npu -j2

# Smoke test
./build-npu/ggnpu bench-matmul

# Inference (put GGUF files in models/)
./build-npu/ggnpu \
  -m models/llama-3.2-1b-q4_k_m.gguf -p "The capital of France is" -c 2048 -n 32
```

### Build NPU kernels (Triton-XDNA)

`bench-matmul` and inference need `.xclbin` kernel files **and their `*_sequence.bin` instruction files** in `~/.cache/ggnpu/xclbin/`. Both are produced by `scripts/build-kernels.sh`; if you copy kernels between machines, copy both files per op.
No NPU hardware is needed to build kernels — compilation runs entirely on CPU.
You can build on any Ubuntu 24.04+ machine (including WSL) with ~16 GB RAM
(32 GB recommended) and copy the `.xclbin` files back to the NPU machine.

#### Step-by-step: build kernels on another machine (native Ubuntu or WSL)

**Step 0 (WSL only) — give WSL enough RAM and clone on the Linux filesystem.**
WSL caps memory at 50% of host RAM by default; kernel compilation needs ~16 GB.
In Windows, edit `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=24GB
swap=16GB
```

Then run `wsl --shutdown` in PowerShell and reopen WSL.

The repo **must** live on the Linux filesystem (e.g. `~/ggnpu`), **not** `/mnt/c/...`.
Builds on Windows mounts fail due to broken symlinks and slow I/O.

**Step 1 — system packages.** `python3-dev` is mandatory; the kernel compiler
links a small launcher against `Python.h` and fails without it.

```bash
sudo apt update
sudo apt install -y build-essential cmake git python3 python3-venv python3-pip python3-dev uuid-dev
```

**Step 2 — XRT dev headers.** Kernel builds need XRT headers even without an NPU.
Add the AMD PPA and install just the dev packages (no driver/firmware needed):

```bash
sudo add-apt-repository -y ppa:amd-team/xrt
sudo apt update
sudo apt install -y libxrt2 libxrt-dev libxrt-utils uuid-dev
```

If the PPA isn't available, `./scripts/build-kernels.sh` will try
`bash scripts/fetch-xrt-dev.sh` automatically, but that also requires `libxrt2`
to be installed — so prefer the PPA route above.

**Step 3 — clone and create the Triton-XDNA venv.** Triton-XDNA is distributed
as GitHub release wheels, not on PyPI, so plain `pip install triton-xdna` fails.
Use the setup script, which passes the required `--find-links` URLs:

```bash
git clone https://github.com/michaelstaake/ggnpu.git ~/ggnpu
cd ~/ggnpu
bash scripts/setup-triton-env.sh          # creates ~/triton-env
```

The script ends by verifying the install; you should see `NPUDriver: OK`.

(If you must install manually instead:)

```bash
python3 -m venv ~/triton-env
source ~/triton-env/bin/activate
pip install --upgrade pip wheel
pip install triton-xdna \
  --find-links https://github.com/amd/Triton-XDNA/releases/expanded_assets/latest-wheels \
  --find-links https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti-2 \
  --find-links https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly \
  --find-links https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Step 4 — preflight check.** This catches every known WSL/venv problem
(Windows-mount repo, missing `python3-dev`, missing XRT headers, link failures)
before you waste time on a long compile:

```bash
source ~/triton-env/bin/activate
bash scripts/verify-kernel-build.sh
```

Fix any `[FAIL]` items it reports, then re-run until it says `Preflight OK`.

**Step 5 — build the kernels.**

```bash
source ~/triton-env/bin/activate
./scripts/build-kernels.sh npu6          # all kernels for npu6 (Krackan)
# ./scripts/build-kernels.sh npu6 matmul # single kernel
# ./scripts/build-kernels.sh npu4 npu5 npu6
```

Output goes to `~/.cache/ggnpu/xclbin/`. Already-built xclbins are skipped, so
re-running after a failure is safe.

**Step 6 — copy the kernels back to the NPU machine.**

```bash
scp -r ~/.cache/ggnpu/xclbin/* user@npu-machine:~/.cache/ggnpu/xclbin/
```

(Or copy via USB/shared folder — anything that lands the `.xclbin` files in
`~/.cache/ggnpu/xclbin/` on the NPU machine works.)

#### Common failures

| Symptom | Cause / fix |
|---------|-------------|
| `ERROR: Triton-XDNA not installed` | venv not activated, or installed without `--find-links` URLs. Run `bash scripts/setup-triton-env.sh` then `source ~/triton-env/bin/activate`. |
| `Python.h` not found / launcher link fails | `sudo apt install python3-dev` |
| `XRT development files not found` | Step 2: install `libxrt2 libxrt-dev uuid-dev` from the AMD PPA. |
| `xclbinutil not found, skipping xclbin generation` | `sudo apt install libxrt-utils libxrt-utils-npu` (or `bash scripts/fetch-xrt-dev.sh --tools`). |
| Build killed / OOM in WSL | Raise `memory=` in `.wslconfig` (Step 0), then `wsl --shutdown`. |
| Symlink or random I/O errors in WSL | Repo is on `/mnt/c` — re-clone to `~/ggnpu`. |
| Wheels not found for your Python | Use Ubuntu's default `python3` (3.10+); avoid exotic interpreters. |

### Verify host

```bash
bash scripts/setup-host.sh
bash scripts/verify-npu.sh
```

These scripts check hardware, driver, permissions, XRT, and the local xclbin cache.

---

**Implementation spec:** [IMPLEMENTATION.md](IMPLEMENTATION.md) — complete handoff for contributors and AI agents.

**Benchmark log:** [dev-benchmark.md](dev-benchmark.md) — running record of timed/correctness runs. **Append a new entry (commit, date, command, result) every time you run a benchmark or timed test.**

**Full host setup guide:** [docs/host-setup-guide.md](docs/host-setup-guide.md)

**Kernel builds:** Local kernel compilation requires `Triton-XDNA`. If you already have prebuilt `.xclbin` files, place them under `~/.cache/ggnpu/xclbin/`.
