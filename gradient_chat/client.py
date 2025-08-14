import json
import requests
from pathlib import Path
from datetime import datetime
from .conversation import GradientConversation
from .headers import generate_headers


class GradientChatClient:
    BASE_URL = "https://chat.gradient.network/api"
    DEFAULT_CONTEXT_SIZE = 15  # 15 Q&As = 30 messages
    MAX_CONTEXT_SIZE = 50      # 25 Q&As = 50 messages (hard cap)

    def __init__(self, model="GPT OSS 120B", cluster_mode="nvidia", log_dir="logs"):
        self.model = model
        self.cluster_mode = cluster_mode
        self.log_base_dir = Path(log_dir)
        self.run_dir = self.log_base_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Text log file for appended conversation
        self.text_log_file = self.run_dir / "conversation_log.txt"

        # Load available models
        self.available_models = self.get_model_info()

        # Internal conversation
        self._internal_conversation = GradientConversation()

        # Headers
        self.headers = generate_headers()

    def get_model_info(self):
        resp = requests.get(f"{self.BASE_URL}/model_info", headers=self.headers)
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            return data.get("availableModels", [])
        return []

    def generate(
        self, 
        user_message: str, 
        context_size: int = None, 
        enableThinking: bool = False,
        model: str = None,
        cluster_mode: str = None,
        conversation: GradientConversation = None
    ):
        # Enforce context size
        if context_size is None or context_size < 0:
            context_size = self.DEFAULT_CONTEXT_SIZE
        context_size = min(context_size, self.MAX_CONTEXT_SIZE)

        # Use provided model and cluster mode or default to self.model and self.cluster_mode
        req_model = model or self.model
        req_cluster_mode = cluster_mode or self.cluster_mode

        # Use provided conversation or internal one
        if conversation is None:
            conversation = self._internal_conversation

        conversation.add_user_message(user_message)

        payload = {
            "model": req_model,
            "clusterMode": req_cluster_mode,
            "messages": conversation.get_context(context_size) or [{"role": "user", "content": user_message}], # if no context, add user message
            "enableThinking": enableThinking
        }

        resp = requests.post(f"{self.BASE_URL}/generate", headers=self.headers, data=json.dumps(payload))
        if resp.status_code != 200:
            raise RuntimeError(f"Request failed: {resp.status_code}")

        raw_lines = resp.text.splitlines()
        reply_content, reasoning_content = [], []
        job_completed = False
        model_used = None

        for line in raw_lines:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            typ, d = data.get("type"), data.get("data", {})
            if typ == "jobInfo" and d.get("status") == "completed":
                job_completed = True
            elif typ == "clusterInfo":
                model_used = d.get("model", model_used)
            elif typ == "reply":
                if d.get("content"): reply_content.append(d["content"])
                if d.get("reasoningContent"): reasoning_content.append(d["reasoningContent"])

        if not job_completed:
            raise RuntimeError("Job did not complete successfully")

        reply_text = "".join(reply_content).strip()
        reasoning_text = "".join(reasoning_content).strip()

        # Update conversation
        conversation.add_assistant_message(reply_text, reasoning_text)

        # Save JSON log
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        json_file = self.run_dir / f"{timestamp}.json"
        with json_file.open("w", encoding="utf-8") as f:
            json.dump({"request": payload, "response": raw_lines}, f, ensure_ascii=False, indent=2)

        # Append text log
        with self.text_log_file.open("a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] Model: {model_used}\n")
            f.write(f"Question: {user_message}\n")
            f.write(f"Reasoning: {reasoning_text}\n")
            f.write(f"Reply: {reply_text}\n")
            f.write("\n" + "-"*50 + "\n\n")

        return {"reply": reply_text, "reasoning": reasoning_text, "model": model_used}

    def get_conversation(self) -> GradientConversation:
        """
        Returns the internal conversation instance used by the client.
        """
        return self._internal_conversation
