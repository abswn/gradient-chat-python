class GradientConversation:
    def __init__(self, max_history=1000):
        self.messages: list[dict] = []
        self.max_history = max_history # Keep at most 1000 messages

    def add_user_message(self, content: str):
        if self.messages and self.messages[-1]["role"] == "user":
            # Merge with previous user message
            self.messages[-1]["content"] += "\n" + content
        else:
            self.messages.append({"role": "user", "content": content})
        self._trim_history()

    def add_assistant_message(self, content: str, reasoningContent: str = None):
        if self.messages and self.messages[-1]["role"] == "assistant":
            self.messages[-1]["content"] += "\n" + content
            # if current response has reasoning content
            if reasoningContent:
                if "reasoningContent" in self.messages[-1]: # merge if previous response had reasoning content
                    self.messages[-1]["reasoningContent"] += "\n" + reasoningContent
                else:
                    self.messages[-1]["reasoningContent"] = reasoningContent # add if previous response didn't have reasoning content
        # previous message is not an assistant, add normally
        else:
            if reasoningContent:
                self.messages.append({"role": "assistant", "content": content, "reasoningContent": reasoningContent})
            else:
                self.messages.append({"role": "assistant", "content": content})
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
            #   {"role": "assistant", "content": "I'm fine.", "reasoningContent": "..."}
            # ]
        """
        if max_pairs <= 0:
            return []
        msgs = []
        count = 0
        size = len(self.messages)
        for i in range(size-1, 0, -2):
            msgs.append(self.messages[i])
            msgs.append(self.messages[i-1]) # loop goes till index = 1, but this line gets the 0th index
            count += 1
            if count >= max_pairs:
                break
        return list(reversed(msgs))



