from chipformer.rv import asm_agent as agent, ref
from chipformer.rv import spec as S

HELLO = """.text
_start:
    li a0, 65
    li a7, 1
    ecall
    li a0, 0
    li a7, 93
    ecall
"""


def test_reference_output_and_fresh_state():
    a = agent.execute(HELLO)
    assert a["status"] == "halted" and a["stdout"] == "A" and a["exit_code"] == 0
    assert agent.execute(HELLO) == a


def test_errors_and_bounds():
    assert agent.execute("not_an_op x1")["status"] == "assembly_error"
    assert agent.execute('.data\n.space 1000000000')["status"] == "assembly_error"
    assert agent.execute('loop: j loop', max_tokens=10)["status"] == "timeout"
    fault = agent.execute('li a7, 12\necall')
    assert fault['status'] == 'fault' and fault['fault']['cause'] == 'ecall'
    assert agent.execute('.text\n_start: li a0, 0x12345678', max_tokens=1)['status'] == 'timeout'


def test_circuit_trace_must_match():
    def circuit(p, budget):
        return ref.Machine.from_program(p).run(budget + 1)[1:][:budget]
    assert agent.execute(HELLO, circuit=circuit)['backend'] == 'circuit_checked'
    assert agent.execute(HELLO, circuit=lambda p, n: [S.tok('HALT')])['status'] == 'circuit_mismatch'
    assert agent.execute('loop: j loop', max_tokens=7, circuit=circuit)['status'] == 'timeout'


def test_retry_feedback_and_format():
    responses = iter(['no code', '```asm\ninvalid\n```', f'```asm\n{HELLO}```'])
    seen = []
    def generate(messages):
        seen.append(list(messages))
        return next(responses)
    history = agent.attempt_loop('print A', generate, agent.execute)
    assert [h['result']['status'] for h in history] == ['format_error', 'assembly_error', 'halted']
    assert 'assembly_error' in seen[-1][-1]['content']


def test_gguf_cli_retry_and_options(tmp_path, monkeypatch, capsys):
    import json
    import sys
    import types

    path = tmp_path / "writer.gguf"
    path.touch()
    calls = []

    class FakeLlama:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def create_chat_completion(self, **kwargs):
            calls.append(("chat", list(kwargs["messages"]), kwargs["max_tokens"],
                          kwargs["temperature"]))
            answer = "bad" if len([c for c in calls if c[0] == "chat"]) == 1 else (
                f"```asm\n{HELLO}```")
            return {"choices": [{"message": {"content": answer}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(
        Llama=FakeLlama, llama_supports_gpu_offload=lambda: True))
    monkeypatch.setattr(sys, "argv", ["asm_agent", "--task", "print A", "--writer-model",
                                      str(path), "--gpu-layers", "4", "--context-size", "2048",
                                      "--max-new", "99"])
    agent.main()
    result = json.loads(capsys.readouterr().out)
    assert [r["result"]["status"] for r in result] == ["format_error", "halted"]
    assert calls[0] == ("init", {"model_path": str(path), "n_ctx": 2048, "n_gpu_layers": 4,
                                 "verbose": False})
    assert calls[1][2:] == (99, 0.0)
    assert "format_error" in calls[2][1][-1]["content"]


def test_gguf_errors(tmp_path, monkeypatch):
    import sys
    import types
    import pytest

    path = tmp_path / "model.gguf"
    with pytest.raises(ValueError, match="not found"):
        agent.gguf_writer(path, 100)
    path.touch()
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(
        llama_supports_gpu_offload=lambda: False))
    with pytest.raises(RuntimeError, match="no GPU backend"):
        agent.gguf_writer(path, 100, n_gpu_layers=-1)


def test_cpu_model_is_not_replaced_by_writer(monkeypatch, capsys):
    import json
    import sys
    import types
    import chipformer.rv

    cpu = types.SimpleNamespace(eval=lambda: cpu, to=lambda device: cpu)
    torch = types.SimpleNamespace(bfloat16="bf16", cuda=types.SimpleNamespace(
        is_available=lambda: False))
    transformers = types.SimpleNamespace(Qwen3_5ForCausalLM=types.SimpleNamespace(
        from_pretrained=lambda *a, **kw: cpu))
    calls = []
    def run(model, c, machine, budget):
        calls.append(model)
        return machine.run(budget + 1)[1:][:budget]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(chipformer.rv, "core", types.SimpleNamespace(build=lambda: "core"),
                        raising=False)
    monkeypatch.setattr(chipformer.rv, "firmware", types.SimpleNamespace(run=run), raising=False)
    monkeypatch.setattr(agent, "hf_writer", lambda *a: lambda messages: f"```asm\n{HELLO}```")
    monkeypatch.setattr(sys, "argv", ["asm_agent", "--task", "print A", "--writer-model",
                                      "writer", "--cpu-model", "cpu"])
    agent.main()
    result = json.loads(capsys.readouterr().out)
    assert calls == [cpu]
    assert result[0]["result"]["backend"] == "circuit_checked"
