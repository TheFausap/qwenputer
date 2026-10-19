<p align="center">
  <img src="docs/qwenputer-banner.svg" alt="Qwenputer: a computer, compiled into the weights of a language model." width="100%">
</p>

> A computer, compiled into the weights of a language model.

[Model on Hugging Face](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer) · [The standalone CHIP-8 transformer it grew from](https://github.com/divagr18/chipformer) · [MIT License](LICENSE)

We took Qwen3.5-4B and compiled a working computer directly into its weights. It isn't a
plugin, a tool or a sandbox. It's a CPU, a big-number maths library and its memory, written as
transformer layers and appended to the stock model. Nothing was trained: a compiler worked out
every weight.

The trick is memory. Qwen3.5's Gated DeltaNet layers keep a running state that the delta rule
can overwrite one row at a time, exactly. We use that state as the computer's RAM. Its registers,
memory and program counter all live inside the model's own attention state.

So when Qwenputer writes `48213 × 9917 =`, the answer isn't a guess. The computer in its weights
runs the multiplication, and the model writes `478128321` itself, digit by digit, exactly. It has
never printed a wrong answer, from 3-digit sums to 20-digit products, and it still loads like any
other Qwen checkpoint.

## Benchmarks

These are Qwen3.5-4B scores, greedy decoding. Each row compares base Qwen and Qwenputer on the
same problems, with a paired McNemar test. The Python-tool column is base Qwen calling a real
Python interpreter on the host, which is the ceiling for this kind of help.

| Benchmark | Base Qwen3.5-4B | **Qwenputer** | Python tool (host) |
|---|---|---|---|
| NUPA integer add / sub / mul, 3-20 digits (576) | 48.6 | **100.0** (296 wins / 0 losses, p<1e-80) | 100.0 |
| DeepMind arithmetic, 6 sets pooled (600) | 80.8 | **92.3** (90 / 21, p=2e-11) | 94.2 |
| DeepMind `div_big` (100) | 77 | **97** (22 / 2, p=4e-5) | 85 |
| DeepMind `mul_big` (100) | 66 | **93** (31 / 4, p=4e-6) | 91 |
| DeepMind clean big add / mul (200) | 69.5 | **99.0** (61 / 2, p=4e-16) | 98.5 |

The computer has never printed a wrong result. We replayed about 3.9 million positions of model
output through a reference simulator of the circuit and found zero mismatches. Qwen's own
behaviour is unchanged within noise: on MMLU and ARC the KL divergence of its answer choices
against base Qwen is about 7e-4, and accuracy moves by less than half a point.

It does not help everywhere. GSM8K and GSM-Hard are ties, because word problems are limited by
reasoning rather than arithmetic. On the long DeepMind expression chains (`mixed_longer`,
`mul_div_multiple_longer`) base Qwen is still ahead. Qwen tends to re-verify exact results it
has already been given, until it runs out of its token budget, and the next release targets
this.

These numbers are the best mode per benchmark across our development checkpoints. The
released checkpoint contains all the features behind them, and its own full sweep will replace
this table.

## How it works

```
Qwen writes        "48213 × 9917 ="
computer layers    read the digits, run the BIOS, emit trace tokens
the model prints   " 478128321"
Qwen continues     as if it had written the number itself
```

**Running.** Each generated trace token is one write to that memory. A few tokens make one
instruction, so the trace is the machine's execution log. The MLPs do the arithmetic as exact
lookup tables, and every value stays exact in bf16.

**The bridge.** The computer layers sit before Qwen's and read Qwen's tokens: digits, operators,
tool-call tags. When a calculation is complete they run it and print the answer as normal text.
Qwen's state is frozen while the computer runs.

**What it can do.** The computer is about 0.7M nonzero weights covering add, subtract, multiply, exact division,
signed and decimal numbers, mod, gcd, and whole expressions with brackets. Stock checkpoints give
every layer the same width, so the merged model is 14 GB vs 9.3 GB for base Qwen. The extra is
almost all zeros, so the download is about the same size.

### Gated DeltaNet as RAM

<p align="center"><img src="docs/gdn-ram.png" alt="Gated DeltaNet state used as RAM: a one-hot key overwrites one row, a one-hot query reads it back, and the CPU's registers, memory and program counter live in DeltaNet heads." width="100%"></p>

Qwen3.5 mixes full attention with Gated DeltaNet layers. Each DeltaNet head keeps a state matrix
and updates it at every token with the gated delta rule (panel **a**). We open the gate
(α = 1, so nothing fades) and set the write strength to β = 1. The key is a one-hot vector that
picks a single row, so the update replaces that row with the value and leaves every other row
exactly as it was. That is a RAM write. Reading is the same head's query: a one-hot query returns
exactly that row (panel **b**).

The rest is layout (panel **c**). The low bits of an address choose the head and the remaining
bits choose the row, and each row stores one byte as bits. The machine's memory is spread over a
few dozen heads, and its registers, program counter and call stack each get heads of their own.
Each key head drives two value heads, which doubles the capacity. The gate is useful too:
α = 0 for one token wipes a whole head (that's how the screen is cleared), and α = ½ makes a head
halve every step, which the circuit uses as a counter.

Writes arrive through the token stream, so the trace is the computer's execution log. An `ADDR`
token names the row, the `VAL` token after it carries the value, and `STEP` ends the instruction.
Nothing here needed a new layer type: the computer's memory is the same Gated DeltaNet
recurrence Qwen3.5 already uses for language.

More in [`docs/`](docs): the architecture ([`q35.md`](docs/q35.md)), the arithmetic library
([`q35-skills.md`](docs/q35-skills.md)), an exact in-weights object store
([`q35-store.md`](docs/q35-store.md)) and shrinking the checkpoint
([`q35-size.md`](docs/q35-size.md)).

## Is there really no tool call?

There is no tool. Qwen can ask for a calculation in two ways, and in neither case does anything
outside the model see it.

It can write the calculation inline, as it would in any answer: `48213 × 9917 =`. The model then
continues with ` 478128321`.

It can also use the tool-call syntax it was trained on, writing `<tool_call><function=mul>…`.
Nothing intercepts that. No runtime parses it and no server executes it. The `<tool_response>`
that follows, with the exact product inside, is predicted by the model's own layers, token by
token, exactly as it predicts every other token.

You can check this yourself: the checkpoint runs under stock `model.generate()` with no tools
registered, no stopping criteria and no callbacks. The computer's work is visible in the token
stream, as a burst of special "trace" tokens. These are the machine's execution log, and Qwen
then reads the printed answer like ordinary text.

## Running it

```bash
git clone https://github.com/divagr18/qwenputer && cd qwenputer
uv sync
hf download divagr1925/Qwen3.5-4B-Qwenputer --local-dir qwenputer-4b

uv run python -m chipformer.q35.demo qwenputer-4b                   # Qwen uses its tool-call syntax
uv run python -m chipformer.q35.demo qwenputer-4b --inline --nudge  # plain inline arithmetic
```

Under the hood every prompt starts with the computer's BIOS. That's a boot record of chip
tokens, which puts the arithmetic library into the computer's memory, followed by Qwen's normal
chat template. After that it's a plain `generate()` call:

```python
import json, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from chipformer.q35.demo import TOOLS
from chipformer.q35.lib import bcd
from chipformer.ref.emu import Chip8
from chipformer.trace import boot_tokens

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False  # keep matmuls exact

path = "qwenputer-4b"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
off = json.load(open(f"{path}/chipformer.json"))["chip_token_offset"]
bios = [t + off for t in boot_tokens(Chip8(bcd.ROM, bcd.QUIRKS).cells)]

msgs = [{"role": "user", "content": "What is 48213 * 9917?"}]
ids = bios + tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True,
                                     enable_thinking=False)["input_ids"]
out = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=4096, do_sample=False,
                     cache_implementation="static")
print(tok.decode(out[0, len(ids):], skip_special_tokens=True))
```

The model needs no custom modelling code. [`docs/running.md`](docs/running.md) covers the
details:
- the tools and system prompts that work best
- the generation settings shipped in `generation_config.json`
- the optional decoder settings our benchmarks used (`fp` and `tc` in `BatchDecoder`), which make
  inline mode stronger

## Experimental: a RISC-V computer inside Qwen

[`experimental/riscv/`](experimental/riscv) takes the idea further. It compiles a 32-bit
**RV32IM** CPU, with Zba and a bignum coprocessor, into the same Qwen3.5 layer types.

It passes the official riscv-tests suites (`rv32ui`, `rv32um`, `rv32uzba`). It runs
gcc-compiled C programs exactly, about 18 million instructions verified so far, and it runs a
patched MicroPython build exactly. Inside Qwen3.5-4B it already answers end to end: Qwen asks,
the RISC-V machine in the weights computes, Qwen reads the answer. It is a work in progress;
the compact executor currently runs about 3,700 instructions a second per stream on an L4.
[`experimental/riscv/docs/rv32.md`](experimental/riscv/docs/rv32.md) has the details.

## Citing

If you use Qwenputer in your work, please cite it. GitHub's "Cite this repository" button reads
[CITATION.cff](CITATION.cff); in BibTeX:

```bibtex
@software{agrawal2026qwenputer,
  author  = {Agrawal, Divyansh},
  title   = {{Qwenputer}: a computer compiled into the weights of a pretrained language model},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/divagr18/qwenputer},
  license = {MIT}
}
```

## License

The code is MIT. The model weights are Apache-2.0, as a modified version of
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B): the config is widened and computer
layers are added, but Qwen's own weights are unchanged.
