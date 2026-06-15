from vectorrag.memory import ConversationMemory


def test_memory_trims_to_budget():
    mem = ConversationMemory(max_history_tokens=20, model="gpt-4o-mini")
    for i in range(50):
        mem.add("user", f"This is message number {i} with some words.")
    # Should have dropped old turns to stay within budget.
    msgs = mem.as_messages()
    assert len(msgs) < 50
    assert msgs[-1]["content"].endswith("words.")


def test_memory_clear():
    mem = ConversationMemory(max_history_tokens=1000)
    mem.add("user", "hi")
    mem.add("assistant", "hello")
    assert len(mem.as_messages()) == 2
    mem.clear()
    assert mem.as_messages() == []


def test_roles_preserved():
    mem = ConversationMemory(max_history_tokens=1000)
    mem.add("user", "q")
    mem.add("assistant", "a")
    roles = [m["role"] for m in mem.as_messages()]
    assert roles == ["user", "assistant"]
