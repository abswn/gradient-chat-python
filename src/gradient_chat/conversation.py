class GradientConversation:
    def __init__(self, max_history=1000):
        self.messages: list[dict] = []
        self.max_history = max_history # Keep at most 1000 messages

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._trim_history()

    def add_assistant_message(self, content: str, reasoningContent: str = ""):
        self.messages.append({"role": "assistant", "content": content, "reasoningContent": reasoningContent})
        self._trim_history()
    
    def _trim_history(self):
        if len(self.messages) > self.max_history:
            # Keep only the last max_history messages
            self.messages = self.messages[-self.max_history:]

    def get_context(self, max_pairs: int):
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
