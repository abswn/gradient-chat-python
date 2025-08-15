import json
import time
import warnings
import requests
from pathlib import Path
from datetime import datetime
from .conversation import GradientConversation
from .headers import generate_headers

class GradientChatError(RuntimeError):
    """Base exception for all GradientChatClient errors."""
    pass

class GradientChatClient:
    BASE_URL = "https://chat.gradient.network/api"
    DEFAULT_CONTEXT_SIZE = 15  # 15 Q&As = 30 messages
    MAX_CONTEXT_SIZE = 50      # 25 Q&As = 50 messages (hard cap)
    DEFAULT_TIMEOUT = 60       # Timeout for response

    def __init__(self, model="GPT OSS 120B", cluster_mode="nvidia", log_dir="logs", timeout=None):
        self.model = model
        self.cluster_mode = cluster_mode
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.log_base_dir = Path(log_dir)
        self.run_dir = self.log_base_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Text log file for appended conversation
        self.text_log_file = self.run_dir / "conversation_log.txt"

        # Headers for API requests
        self.headers = generate_headers()

        # Load available models
        self.available_models = self.get_model_info()

        # Internal conversation
        self._internal_conversation = GradientConversation()

    def get_model_info(self):
        """Fetch available models. Returns empty list on failure."""
        try:
            resp = requests.get(f"{self.BASE_URL}/model_info", headers=self.headers, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                return data.get("availableModels", [])
        except (requests.exceptions.Timeout, requests.exceptions.RequestException, json.JSONDecodeError):
            return []

    def generate(
        self, 
        user_message: str, 
        context_size: int = None, 
        enableThinking: bool = False,
        model: str = None,
        cluster_mode: str = None,
        timeout: int = None,
        conversation: GradientConversation = None
    ):
        """ Sends user message and gets reply and reasoning

        Args:
            user_message (str): The user message to send.
            context_size (int, optional): Number of previous Q&A pairs to include in context. 
                                          Defaults to DEFAULT_CONTEXT_SIZE.
            enableThinking (bool, optional): Whether to enable step-by-step reasoning. Defaults to False.
            model (str, optional): Model name to use. Defaults to self.model.
            cluster_mode (str, optional): Cluster mode to use. Defaults to self.cluster_mode.
            timeout (int, optional): Timeout for response. Defaults to DEFAULT_TIMEOUT.
            conversation (GradientConversation, optional): Conversation instance. Defaults to internal conversation.

        Returns:
            dict: {"reply": str, "reasoning": str, "model": str}

        Raises:
            GradientChatError: If network fails, request times out, HTTP error occurs, 
                               or inference job did not complete successfully.
                - **Timeout** → The request exceeded `timeout` seconds.
                - **HTTP error** → The server returned a non-2xx status code.
                - **Network error** → Connection issues, DNS failures, etc.
                - **Job failed** → API responded, but job status never reached "completed".
            Warning: If writing JSON or text log fails (non-fatal).
        """
        # Enforce context size
        if context_size is None or context_size < 0:
            context_size = self.DEFAULT_CONTEXT_SIZE
        context_size = min(context_size, self.MAX_CONTEXT_SIZE)

        # Use provided model, cluster mode and timeout or use default values
        req_model = model or self.model
        req_cluster_mode = cluster_mode or self.cluster_mode
        req_timeout = timeout or self.DEFAULT_TIMEOUT

        # Use provided conversation or internal one
        if conversation is None:
            conversation = self._internal_conversation

        # Request payload
        payload = {
            "model": req_model,
            "clusterMode": req_cluster_mode,
            "messages": conversation.get_context(context_size) + [{"role": "user", "content": user_message}], # include current message
            "enableThinking": enableThinking
        }

        # Send request
        try:
            resp = requests.post(f"{self.BASE_URL}/generate", headers=self.headers, data=json.dumps(payload), timeout=req_timeout)
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            raise GradientChatError("Request timeout, please retry")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "unknown"
            text = e.response.text if e.response else ""
            raise GradientChatError(f"HTTP error {status}: {text}")
        except requests.exceptions.RequestException as e:
            raise GradientChatError(f"Network error: {e}")

        # Parse response
        raw_lines = resp.text.splitlines()
        reply_content, reasoning_content = [], []
        job_completed = False
        model_used = None

        for line in raw_lines:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue # skip malformed lines
            typ, d = data.get("type"), data.get("data", {})
            if typ == "jobInfo" and d.get("status") == "completed":
                job_completed = True
            elif typ == "clusterInfo":
                model_used = d.get("model", model_used)
            elif typ == "reply":
                if d.get("content"):
                    reply_content.append(d["content"])
                if d.get("reasoningContent"):
                    reasoning_content.append(d["reasoningContent"])

        if not job_completed:
            raise GradientChatError("Job failed")

        reply_text = "".join(reply_content).strip()
        reasoning_text = "".join(reasoning_content).strip()

        # Update conversation with user's message and assitant's reply
        conversation.add_user_message(user_message)
        if reasoning_text:
            conversation.add_assistant_message(reply_text, reasoning_text)
        else:
            conversation.add_assistant_message(reply_text)

        # Save JSON log
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        json_file = self.run_dir / f"{timestamp}.json"
        try:
            with json_file.open("w", encoding="utf-8") as f:
                json.dump({"request": payload, "response": raw_lines}, f, ensure_ascii=False, indent=2)
        except OSError as e:
            warnings.warn(f"Failed to write JSON log: {e}")

        # Append text log
        try:
            with self.text_log_file.open("a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] Model: {model_used}\n")
                f.write(f"Question: {user_message}\n")
                f.write(f"Reasoning: {reasoning_text}\n")
                f.write(f"Reply: {reply_text}\n")
                f.write("\n" + "-"*50 + "\n\n")
        except OSError as e:
            warnings.warn(f"Failed to write text log: {e}")

        return {"reply": reply_text, "reasoning": reasoning_text, "model": model_used}

    def get_conversation(self) -> GradientConversation:
        """Returns the internal conversation instance used by the client."""
        return self._internal_conversation
