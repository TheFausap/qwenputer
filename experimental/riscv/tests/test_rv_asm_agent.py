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
