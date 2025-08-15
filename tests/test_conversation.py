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

def test_merge__repeated_assistant_messages():
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
