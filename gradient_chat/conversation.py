class GradientConversation:
    def __init__(self):
        self.messages: list[dict] = []

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str, reasoningContent: str = ""):
        self.messages.append({"role": "assistant", "content": content, "reasoningContent": reasoningContent})

    def get_context(self, max_pairs: int = 5):
        """
        Returns the most recent conversation context containing up to `max_pairs` assistant responses 
        and their corresponding user messages, in chronological order.

        Example:
            convo = GradientConversation()
            convo.add_user_message("Hi")
            convo.add_assistant_message("Hello!")
            convo.add_user_message("How are you?")
            convo.add_assistant_message("I'm fine.")

            convo.get_context(max_pairs=1)
            # Returns:
            # [
            #   {"role": "user", "content": "How are you?"},
            #   {"role": "assistant", "content": "I'm fine.", "reasoningContent": ""}
            # ]
        """
        if max_pairs <= 0:
            return []
        msgs = []
        count = 0
        for msg in reversed(self.messages):
            msgs.append(msg)
            if msg["role"] == "assistant":
                count += 1
            if count >= max_pairs:
                break
        return list(reversed(msgs))
