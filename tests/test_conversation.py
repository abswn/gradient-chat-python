import pytest
from gradient_chat.conversation import GradientConversation


def test_add_user_message():
    convo = GradientConversation()
    convo.add_user_message("Hello")
    assert len(convo.messages) == 1
    assert convo.messages[0]["role"] == "user"
    assert convo.messages[0]["content"] == "Hello"


def test_add_assistant_message_without_reasoning():
    convo = GradientConversation()
    convo.add_assistant_message("Hi there")
    assert len(convo.messages) == 1
    msg = convo.messages[0]
    assert msg["role"] == "assistant"
    assert msg["content"] == "Hi there"
    assert "reasoningContent" not in msg


def test_add_assistant_message_with_reasoning():
    convo = GradientConversation()
    convo.add_assistant_message("Answer", reasoningContent="Logic here")
    msg = convo.messages[0]
    assert msg["role"] == "assistant"
    assert msg["content"] == "Answer"
    assert msg["reasoningContent"] == "Logic here"


def test_merge_repeated_user_messages():
    convo = GradientConversation()
    convo.add_user_message("Hi")
    convo.add_user_message("How are you?")
    assert convo.messages[-1]["content"] == "Hi\nHow are you?"


def test_merge_repeated_assistant_messages():
    convo = GradientConversation()
    convo.add_assistant_message("Hello!")
    convo.add_assistant_message("I am fine.", "Reasoning1")
    last_msg = convo.messages[-1]
    assert last_msg["content"] == "Hello!\nI am fine."
    assert last_msg["reasoningContent"] == "Reasoning1"


def test_trim_history():
    max_hist = 3
    convo = GradientConversation(max_history=max_hist)
    for i in range(5):
        convo.add_user_message(f"user {i}")
        convo.add_assistant_message(f"assistant {i}")
    assert len(convo.messages) <= max_hist


def test_get_context_basic():
    convo = GradientConversation()
    convo.add_user_message("Hi")
    convo.add_assistant_message("Hello!")
    convo.add_user_message("How are you?")
    convo.add_assistant_message("I'm fine.")
    context = convo.get_context(max_pairs=1)
    assert len(context) == 2
    assert context[0]["role"] == "user"
    assistant_msg = context[1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == "I'm fine."
    # Only check reasoningContent if it exists
    if "reasoningContent" in assistant_msg:
        assert isinstance(assistant_msg["reasoningContent"], str)


def test_get_context_zero_pairs():
    convo = GradientConversation()
    convo.add_user_message("Hi")
    convo.add_assistant_message("Hello!")
    context = convo.get_context(max_pairs=0)
    assert context == []


def test_get_context_more_than_available():
    convo = GradientConversation()
    convo.add_user_message("Hi")
    convo.add_assistant_message("Hello!")
    context = convo.get_context(max_pairs=5)
    # Should return all messages
    assert len(context) == 2
    assert context[0]["role"] == "user"
    assert context[1]["role"] == "assistant"


def test_add_user_message_new_and_merge():
    convo = GradientConversation()
    convo.add_user_message("Hello")
    assert len(convo.messages) == 1
    assert convo.messages[0]["role"] == "user"
    assert convo.messages[0]["content"] == "Hello"

    # Merge with previous user message
    convo.add_user_message("How are you?")
    assert len(convo.messages) == 1
    assert convo.messages[0]["role"] == "user"
    assert convo.messages[0]["content"] == "Hello\nHow are you?"


def test_add_assistant_message_without_reasoning_then_merge_without_reasoning():
    convo = GradientConversation()
    convo.add_assistant_message("Hi there")
    assert len(convo.messages) == 1
    msg = convo.messages[0]
    assert msg["role"] == "assistant"
    assert msg["content"] == "Hi there"
    assert "reasoningContent" not in msg

    # Merge next assistant message content
    convo.add_assistant_message("I can help.")
    assert len(convo.messages) == 1
    msg = convo.messages[0]
    assert msg["role"] == "assistant"
    assert msg["content"] == "Hi there\nI can help."
    assert "reasoningContent" not in msg


def test_add_assistant_message_with_reasoning_then_merge_reasoning():
    # First assistant has reasoning
    convo = GradientConversation()
    convo.add_assistant_message("Answer 1", reasoningContent="Reason 1")
    assert len(convo.messages) == 1
    msg = convo.messages[0]
    assert msg["role"] == "assistant"
    assert msg["content"] == "Answer 1"
    assert msg["reasoningContent"] == "Reason 1"

    # Merge both content and reasoning into the same last assistant message
    convo.add_assistant_message("Answer 2", reasoningContent="Reason 2")
    msg = convo.messages[0]
    assert msg["role"] == "assistant"
    assert msg["content"] == "Answer 1\nAnswer 2"
    assert msg["reasoningContent"] == "Reason 1\nReason 2"


def test_sequence_user_assistant_user_assistant_context_ordering():
    convo = GradientConversation()
    convo.add_user_message("U1")
    convo.add_assistant_message("A1")
    convo.add_user_message("U2")
    convo.add_assistant_message("A2", reasoningContent="R2")

    # Latest single pair: [U2, A2]
    ctx = convo.get_context(max_pairs=1)
    assert len(ctx) == 2
    assert ctx[0] == {"role": "user", "content": "U2"}
    assert ctx[1]["role"] == "assistant"
    assert ctx[1]["content"] == "A2"
    if "reasoningContent" in ctx[1]:
        assert ctx[1]["reasoningContent"] == "R2"

    # Two pairs: [U1, A1, U2, A2]
    ctx2 = convo.get_context(max_pairs=2)
    assert len(ctx2) == 4
    assert ctx2[0] == {"role": "user", "content": "U1"}
    assert ctx2[1] == {"role": "assistant", "content": "A1"}
    assert ctx2[2] == {"role": "user", "content": "U2"}
    assert ctx2[3]["role"] == "assistant"
    assert ctx2[3]["content"] == "A2"


def test_consecutive_assistant_messages_merge_correctly_then_user_breaks_merge():
    convo = GradientConversation()
    convo.add_assistant_message("A1")
    convo.add_assistant_message("A2")
    assert len(convo.messages) == 1
    assert convo.messages[0]["content"] == "A1\nA2"

    # User message breaks assistant merge sequence
    convo.add_user_message("U1")
    convo.add_assistant_message("A3")
    assert len(convo.messages) == 3
    assert convo.messages[0]["role"] == "assistant"
    assert convo.messages[0]["content"] == "A1\nA2"
    assert convo.messages[1]["role"] == "user"
    assert convo.messages[1]["content"] == "U1"
    assert convo.messages[2]["role"] == "assistant"
    assert convo.messages[2]["content"] == "A3"
