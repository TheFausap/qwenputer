"""Host-orchestrated assembly generation and execution (see experimental/riscv/README.md)."""

import argparse
import json
import re
from pathlib import Path

from chipformer.rv import asm, ref
from chipformer.rv import spec as S

MAX_SOURCE = 16000
SYSTEM = """Write a complete program for a small RV32IM machine, in the assembly subset below.
Return exactly one ```asm fenced block, with no other code blocks. The host assembles and runs it;
if you receive a failure, correct the program and return a complete replacement block.
Entry is _start (or the first instruction). Registers start at zero except sp = 0x8000.
Instructions are RV32IM and Zba with standard register names and these pseudos: li, la, mv,
not, neg, seqz, snez, nop, j, jr, ret, call, tail, beqz, bnez, blez, bgez, bltz, bgtz,
bgt, ble, bgtu, bleu. Labels and .text, .data, .rodata, .word, .half, .byte, .globl are
supported. No .space or .align. RAM is 0..0x7fff; .data starts at 0x100. Code is at 0x40000,
.rodata at 0x20000. Loads and stores must be aligned. No OS, standard library or stdin.
To print one byte: put it in a0, set a7 to 1, and ecall. To finish: set a0 to the exit code,
a7 to 93, and ecall. You must explicitly exit. Output and exit code are the only results.
"""


def extract_assembly(text):
    blocks = re.findall(r"```(?:asm|riscv|assembly)\s*\n(.*?)```", text, re.I | re.S)
    if len(blocks) != 1:
        raise ValueError("expected exactly one fenced asm block")
    return blocks[0]


def prepare(source):
    if len(source.encode("utf-8")) > MAX_SOURCE:
        raise ValueError("source exceeds 16000 bytes")
    # The test assembler's size directives allocate directly, including on pass one. Reject them
    # rather than allowing model-generated source to cause unbounded host allocations.
    for line in source.splitlines():
        line = re.split(r"[;#]", line)[0]
        if re.search(r"(?<![\w.$])\.(?:space|align)\b", line, re.I):
            raise ValueError(".space and .align are not supported")
    try:
        p = asm.assemble(source)
    except (ValueError, KeyError, IndexError, AssertionError, OverflowError, TypeError) as e:
        raise ValueError(f"assembly error: {e or type(e).__name__}") from e
    if not p.text or len(p.text) > S.ROM_WORDS:
        raise ValueError("program must contain code that fits ROM")
    if not S.ROM_BASE <= p.entry < S.ROM_BASE + 4 * len(p.text) or p.entry % 4:
        raise ValueError("entry must point to an aligned instruction in the program")
    if p.data_base < S.RAM_BASE or p.data_base + len(p.data) > S.RAM_BASE + 4 * S.RAM_WORDS:
        raise ValueError("data does not fit RAM")
    if len(p.rodata) > 4 * S.DROM_WORDS:
        raise ValueError("rodata does not fit DROM")
    return p


def execute(source, max_tokens=4096, circuit=None):
    """Return bounded execution feedback. circuit(model, program, budget) returns chip tokens.

    The reference trace is also used to check every generated circuit token; it does not supply
    the circuit's output. A fresh state is used for every execution.
    """
    try:
        p = prepare(source)
    except ValueError as e:
        return {"status": "assembly_error", "error": str(e)}
    m = ref.Machine.from_program(p)
    expected = []
    incomplete = False
    try:
        while not m.halted and len(expected) < max_tokens:
            previous = (m.steps, len(m.out))
            record = m.record()
            remaining = max_tokens - len(expected)
            expected.extend(record[:remaining])
            if len(record) > remaining:
                # The record commits only on its final token, outside the budget.
                m.steps, out_len = previous
                del m.out[out_len:]
                incomplete = True
                break
    except (IndexError, ValueError) as e:
        return {"status": "reference_error", "error": str(e)}
    if circuit is not None:
        got = circuit(p, max_tokens)
        if got != expected:
            first = next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b),
                         min(len(got), len(expected)))
            return {"status": "circuit_mismatch", "first_token": first,
                    "generated_tokens": len(got), "reference_tokens": len(expected)}
    result = {"status": "timeout" if incomplete or not m.halted else "fault" if m.fault else "halted",
              "stdout": bytes(m.out[:512]).decode("latin-1"),
              "stdout_truncated": len(m.out) > 512, "exit_code": None if incomplete else m.exit_code,
              "instructions": m.steps, "trace_tokens": len(expected),
              "backend": "circuit_checked" if circuit is not None else "reference"}
    if m.fault and not incomplete:
        result["fault"] = {"cause": S.FAULTS[m.fault[0]], "pc": hex(m.fault[1])}
    return result


def attempt_loop(task, generate, run, attempts=3):
    """Return attempts; execution feedback is supplied as a user message on each retry."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    history = []
    for _ in range(attempts):
        response = generate(messages)
        messages.append({"role": "assistant", "content": response})
        try:
            source = extract_assembly(response)
            result = run(source)
        except ValueError as e:
            source, result = None, {"status": "format_error", "error": str(e)}
        history.append({"response": response, "source": source, "result": result})
        # Execution success does not prove task correctness. Let the user inspect the output;
        # only errors trigger automatic retries in this baseline.
        if result["status"] == "halted" and result["exit_code"] == 0:
            break
        messages.append({"role": "user", "content": "Execution feedback: " + json.dumps(result)
                         + "\nPlease fix the program and try again."})
    return history


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", help="run assembly from a file (no language model needed)")
    group.add_argument("--task", help="task for a language model to implement")
    ap.add_argument("--writer-model", help="local HF causal language model or model ID for --task")
    ap.add_argument("--cpu-model", help="local rv-tiny checkpoint; otherwise use reference only")
    ap.add_argument("--max-tokens", type=int, default=4096, help="CPU trace token budget")
    ap.add_argument("--max-new", type=int, default=1024, help="writer generation token budget")
    ap.add_argument("--attempts", type=int, default=3)
    args = ap.parse_args()
    if min(args.max_tokens, args.max_new, args.attempts) < 1:
        ap.error("budgets and attempts must be positive")
    if args.task and not args.writer_model:
        ap.error("--task requires --writer-model")
    circuit = None
    if args.cpu_model:
        import torch
        from transformers import Qwen3_5ForCausalLM
        from chipformer.rv import core, firmware

        model = Qwen3_5ForCausalLM.from_pretrained(args.cpu_model, dtype=torch.bfloat16).eval()
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        c = core.build()
        circuit = lambda p, budget: firmware.run(model, c, ref.Machine.from_program(p), budget)
    run = lambda source: execute(source, args.max_tokens, circuit)
    if args.source:
        print(json.dumps(run(Path(args.source).read_text(encoding="utf-8")), indent=2))
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.writer_model)
    model = AutoModelForCausalLM.from_pretrained(args.writer_model, dtype=torch.bfloat16).eval()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    def generate(messages):
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                      return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new, do_sample=False)
        return tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)

    print(json.dumps(attempt_loop(args.task, generate, run, args.attempts), indent=2))


if __name__ == "__main__":
    main()
