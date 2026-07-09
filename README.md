# ggnpu

Run standard GGUF models on AMD NPUs (Krackan / XDNA2). No conversion or
re-quantization — point `ggnpu` at a `.gguf` file and it runs.

## Supported models & quantizations

**Validated on Krackan (`npu6`):** Llama 3.2 1B, Qwen2.5 / Qwen2.5-Coder 1.5B,
DeepSeek-R1-Distill-Qwen 1.5B, Qwen3 0.6B, Qwen3.5 (Gated DeltaNet hybrid),
Qwen3.6-35B-A3B (`qwen35moe`, MoE + shared expert), Gemma 3n E2B, LFM2.5 230M.
Other `llama`- and `qwen2`-family GGUFs generally work through the same path.

**Quantizations:** `Q2_K` through `Q8_0`, `Q4_K`/`Q5_K`/`Q6_K` variants,
`IQ4_NL`, `IQ4_XS`, `BF16`, and the legacy ARM-repacked `Q4_0_4_4` (de-interleaved
to `Q4_0` on load). Mixed-precision GGUFs (e.g. `Q4_K_M` with `Q6_K` attention
layers) are handled tensor-by-tensor. Full details:
[IMPLEMENTATION.md §1.2](IMPLEMENTATION.md#12-model-and-quantization-support).

## What you need

| Item | Notes |
|------|-------|
| AMD Ryzen AI system | Krackan or Strix Point with NPU enabled in BIOS |
| Ubuntu 24.04 or 26.04 | Native Linux, or a Proxmox LXC container (see below) — not Docker |
| ~4 GB disk | Plus space for model files |
| A GGUF model | e.g. Llama 3.2 1B Q4_K_M |

**BIOS:** NPU/IPU enabled. **Kernel:** `amd_iommu=on` at boot.

### Running inside a Proxmox LXC container

The NPU works from an unprivileged Ubuntu LXC container, but the passthrough
and limits are configured on the **Proxmox host**, not inside the container.

1. **Enable IOMMU on the host.** Add `amd_iommu=on iommu=pt` to the kernel
   command line (`/etc/default/grub` → `GRUB_CMDLINE_LINUX_DEFAULT`, then
   `update-grub`, or edit `/etc/kernel/cmdline` for systemd-boot). Reboot.

2. **Pass the NPU device into the container.** In Proxmox UI, Container > Resources > Add > Device Passthrough. Should be something like /dev/accel/accel0.

3. **Raise the memlock limit for the container.** The NPU driver locks large
   buffers; the default 8 MB cap makes device open fail with
   `mmap(...) failed (err=-11): Resource temporarily unavailable`. Add to the
   same config file:

   ```
   lxc.prlimit.memlock: unlimited
   ```

4. **Restart the container** (`pct restart <vmid>`) so the changes apply.

Inside the container, install the `amdxdna` driver / XRT as usual (`./setup.sh`
handles this). `bash scripts/verify-npu.sh` may report `[FAIL]` for **IOMMU**
and **NPU firmware** even when everything works — those checks read host
`/sys` nodes the container can't see; `./build-npu/ggnpu bench-matmul` is the
real test.

## Quick start

### Step 1 — Clone and set up

```bash
git clone https://github.com/michaelstaake/ggnpu.git
cd ggnpu
./setup.sh
```

`./setup.sh` installs host dependencies, builds `ggnpu`, and walks you through
building NPU kernels. It is interactive and safe to re-run — finished steps are
skipped.

**After setup:** log out and back in (or reboot) if the script added you to the
`render` group or changed memlock limits.

Useful flags:

| Flag | Description |
|------|-------------|
| `--yes` | Non-interactive (accept defaults) |
| `--kernels-here` | Build NPU kernels on this machine |
| `--kernels-separate` | Skip kernels (you will build on another machine) |
| `--no-kernels` | Skip kernel build entirely |
| `--cpu` | CPU-only build (no NPU hardware needed) |
| `--skip-install` | Skip privileged host install |
| `--skip-build` | Skip the C++ build |

### Step 2 — Build NPU kernels

Inference and `bench-matmul` need compiled kernel files in
`~/.cache/ggnpu/xclbin/`. Each operation needs **both** a `.xclbin` and its
matching `*_sequence.bin` file.

Kernel compilation runs on **CPU only** — no NPU hardware is required to build.
It needs **~16 GB RAM** (32 GB recommended) and takes a while.

Pick one of the two options below.

#### Option A — Build on the NPU host

Use this when your NPU machine has enough RAM (~16 GB+).

During `./setup.sh`, choose **build kernels here**, or run:

```bash
./setup.sh --kernels-here
```

Or manually:

```bash
bash scripts/setup-triton-env.sh       # one-time: creates ~/triton-env
source ~/triton-env/bin/activate
bash scripts/verify-kernel-build.sh    # fix any [FAIL] before compiling
./scripts/build-kernels.sh npu6        # all kernels for Krackan
```

Already-built kernels are skipped, so re-running after a failure is safe.

#### Option B — Build on a separate machine

Use this when the NPU host does **not** have enough RAM for kernel compilation.
Any Ubuntu 24.04+ PC can build the kernels — no NPU, driver, or firmware needed.

**On the build machine:**

```bash
# 1. System packages (python3-dev is required)
sudo apt update
sudo apt install -y build-essential cmake git python3 python3-venv python3-pip python3-dev uuid-dev

# 2. XRT dev headers (no driver needed)
sudo add-apt-repository -y ppa:amd-team/xrt
sudo apt update
sudo apt install -y libxrt2 libxrt-dev libxrt-utils

# 3. Clone, set up Triton, and build
git clone https://github.com/michaelstaake/ggnpu.git ~/ggnpu
cd ~/ggnpu
bash scripts/setup-triton-env.sh
source ~/triton-env/bin/activate
bash scripts/verify-kernel-build.sh
./scripts/build-kernels.sh npu6
```

**WSL users:** raise memory in `C:\Users\<you>\.wslconfig` (`memory=24GB`,
`swap=16GB`), run `wsl --shutdown`, then reopen WSL. Clone to the Linux
filesystem (`~/ggnpu`), **not** `/mnt/c/...`.

**Copy kernels to the NPU host:**

```bash
scp -r ~/.cache/ggnpu/xclbin/* user@npu-host:~/.cache/ggnpu/xclbin/
```

Copy both `.xclbin` and `*_sequence.bin` files for each operation.

If you already have prebuilt kernels, place them directly in
`~/.cache/ggnpu/xclbin/` on the NPU host.

### Step 3 — Verify

```bash
bash scripts/verify-npu.sh
./build-npu/ggnpu bench-matmul
```

### Step 4 — Run inference

Put your `.gguf` model in `models/`, then:

```bash
./build-npu/ggnpu \
  -m models/llama-3.2-1b-q4_k_m.gguf \
  -p "The capital of France is" \
  -c 2048 \
  -n 32
```

More examples and CLI flags: [docs/usage.md](docs/usage.md).

## Common kernel build failures

| Symptom | Fix |
|---------|-----|
| `Triton-XDNA not installed` | Run `bash scripts/setup-triton-env.sh`, then `source ~/triton-env/bin/activate` |
| `Python.h` not found | `sudo apt install python3-dev` |
| `XRT development files not found` | `sudo apt install libxrt2 libxrt-dev uuid-dev` from the AMD PPA |
| `xclbinutil not found` | `sudo apt install libxrt-utils` |
| Build killed / OOM | Use **Option B** (build on another machine) or add RAM |
| Symlink or I/O errors in WSL | Re-clone to `~/ggnpu`, not `/mnt/c/...` |

## Manual setup (alternative to `./setup.sh`)

<details>
<summary>What <code>setup.sh</code> automates</summary>

```bash
# 1. Core build tools + XRT/NPU driver
bash scripts/install-host.sh

# 2. Build ggnpu
./scripts/build.sh

# 3. Verify host
bash scripts/setup-host.sh
bash scripts/verify-npu.sh
```

See [docs/host-setup-guide.md](docs/host-setup-guide.md) for the full manual
walkthrough including memlock limits, render group, and IOMMU.

</details>

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/usage.md](docs/usage.md) | CLI reference, flags, examples |
| [docs/host-setup-guide.md](docs/host-setup-guide.md) | Detailed host setup and troubleshooting |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | Architecture, supported models, contributor handoff |
| [dev-benchmark.md](dev-benchmark.md) | Benchmark and correctness log |
