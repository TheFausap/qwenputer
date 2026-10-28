# A RISC-V computer inside Qwen

Model: [divagr1925/Qwen3.5-4B-Qwenputer](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer) on Hugging Face, or [Spartan1925/Qwen3.5-4B-Qwenputer](https://www.modelscope.ai/models/Spartan1925/Qwen3.5-4B-Qwenputer) on ModelScope.

This folder compiles a 32-bit **RV32IM** CPU into stock Qwen3.5 layers, using the same compiler as
the main project. The CPU also has Zba and a bignum coprocessor. It is a self-contained snapshot:
it carries its own copy of the `chipformer` package, because its compiler changes haven't been
merged with the main tree yet.

## What works

- The official riscv-tests suites `rv32ui`, `rv32um` and `rv32uzba` pass. They are checked against
  the reference model and against qemu.
- gcc-compiled C runs exactly: crc32, qsort, a sieve, mini-gmp and a fuzz suite, about 18M
  instructions checked free-running. A patched MicroPython build also runs exactly.
- A `calc` firmware with bignum `+ - * / // % ** gcd` is exact on 10,000 random cases.
- `rv-tiny`, a standalone stock `Qwen3_5ForCausalLM`, runs programs from a memory snapshot under
  plain `generate()`.
- Inside Qwen3.5-4B, Qwen calls `calc` and the RISC-V machine in the weights answers. A replay of
  the evaluation records found 0 mismatches.

The machine has 256 KB of code ROM, 128 KB of read-only data and 32 KB of RAM, all held in Gated
DeltaNet state. It takes 11 chip layers and about 1M nonzero weights, and costs about 1.2-1.4
tokens per instruction. A compact executor runs about 3,700 instructions a second per stream on
an L4, and in generation it runs at about 3,200 trace tokens a second.

[`docs/rv32.md`](docs/rv32.md) has the design and every measurement, and
[`docs/rv32-p3-status.md`](docs/rv32-p3-status.md) has the in-Qwen status.

```bash
cd experimental/riscv && uv sync --extra hf && uv run pytest tests -q
```

## Host-side assembly writer prototype

`chipformer.rv.asm_agent` lets a local language model write assembly, then assembles it on the
host and runs a **fresh** RV32 machine per attempt. Faults, timeouts and assembly errors are fed
back to the writer for up to three attempts. This is orchestration, not an in-model assembler.
It uses the test assembler's limited syntax, not GNU `as`; the complete contract is in
`asm_agent.SYSTEM`. There is no stdin or operating system. A zero exit code only means the
program terminated normally, **not** that it solved the task.

From `experimental/riscv`, with dependencies installed (`uv sync --extra hf`):

```bash
# First try a program without a language model or GPU:
cat > /tmp/print-a.s <<'EOF'
_start:
    li a0, 65
    li a7, 1
    ecall
    li a0, 0
    li a7, 93
    ecall
EOF
uv run python -m chipformer.rv.asm_agent --source /tmp/print-a.s

# Ask an instruction-tuned local Qwen checkpoint to write a complete program:
uv run python -m chipformer.rv.asm_agent --writer-model /path/to/Qwen3.5-4B \
  --task 'Print the decimal digits 123, followed by a newline.'
```

By default execution uses the Python **reference machine**, not the model CPU. For actual
free-running model execution, build a standalone `rv-tiny` checkpoint as described in
`chipformer.rv.tiny`, then add `--cpu-model /path/to/rv-tiny` to either command. The host loads
the assembled program into a fresh circuit snapshot and checks every generated chip token against
a separately run reference trace; a mismatch is reported rather than presented as a result.
This backend uses the standalone CPU model, not the merged Qwenputer chat checkpoint. Loading
both writer and CPU models can require substantial memory. The model backend requires the optional
HF dependencies and has not been optimized for speed.

`--max-tokens` bounds CPU trace tokens, `--max-new` bounds writer output, and `--attempts` bounds
retries. Results are JSON; output is capped at 512 bytes. Model execution and host assembly are
experimental, not a security sandbox. For meaningful evaluation, supply separate test cases and
check the program output against expected results; this prototype retries execution failures but
has no correctness oracle or automatic test-input facility yet.

### Quantized GGUF writer (lower VRAM)

The **writer** can instead be a local GGUF instruction/chat model, using llama.cpp through
`llama-cpp-python`. Use a GGUF whose architecture is supported by the installed llama.cpp and
which contains a suitable chat template (for example a Qwen3.5 instruct GGUF); a base or
incompatible model may not follow the assembly protocol. Obtain the GGUF separately. The model
is loaded from its local file; a Hugging Face repository ID is not a GGUF path.

```bash
# CPU-only llama.cpp build (a C++ compiler and build tools may be needed):
uv sync --extra gguf
uv run --extra gguf python -m chipformer.rv.asm_agent \
  --writer-model /path/to/qwen-instruct-q4.gguf \
  --task 'Print the decimal digits 123, followed by a newline.'
```

For an NVIDIA CUDA build, install the CUDA toolkit and an appropriate compiler, then rebuild
llama-cpp-python with CUDA enabled (this can take time):

```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv sync --extra gguf \
  --no-binary-package llama-cpp-python --reinstall-package llama-cpp-python
uv run --extra gguf python -m chipformer.rv.asm_agent \
  --writer-model /path/to/qwen-instruct-q4.gguf --gpu-layers 8 \
  --context-size 4096 --task 'Print the decimal digits 123, followed by a newline.'
```

`--gpu-layers 0` (the default) keeps the writer on CPU; positive values offload that many layers,
and `-1` requests all layers. The best value depends on available VRAM; context and model
architecture also affect memory. Increase `--context-size` if a task plus retry history exceeds
it, or lower `--attempts`/`--max-new` to reduce memory use. `--writer-backend gguf` forces the
GGUF backend for files without a `.gguf` suffix. The GGUF model's own template is used; thinking
mode and output quality can differ from the HF writer, so inspect results.

This only quantizes the language **writer**. With no `--cpu-model`, execution is still the Python
reference machine. For actual circuit execution, `--cpu-model` still requires the unquantized
`rv-tiny` HF checkpoint and the `hf` dependencies (use `uv sync` and `uv run` with both `--extra gguf --extra hf`); both models
must fit in memory. Use `--cpu-device cpu` to keep the circuit on system RAM instead of the GPU. Quantizing the circuit or the merged Qwenputer checkpoint is not supported:
its exact arithmetic and state behavior need separate validation. These options do not change
`chipformer.rv.chat` or make the merged Qwenputer checkpoint loadable as GGUF.
