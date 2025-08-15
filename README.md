# gradient-chat-python
Unofficial Python client for Gradient Chat which utilizes the decentralized inference network called **Parallax**. When using Gradient Chat (i.e Parallax), the inference load is distributed among multiple P2P devices.

*Note: Currently Parallax is in testing phase and has limited number of participating devices.*

## Features
* Maintain conversation context between requests.
* Optionally choose model, cluster mode and context size per request.
    * GPT OSS 120B
    * Qwen3 235B
* Support for reasoning output (`enableThinking`).
* Logging of all requests and responses (JSON + plain text).

## Installation
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

pip install gradient-chat
```
Or if you want to install the latest development version:
```bash
pip install git+https://github.com/abswn/gradient-chat-python.git
```

## Usage
```python
from gradient_chat import GradientChatClient, GradientChatError

# Create client
client = GradientChatClient()

# Show available models
print("Available Models:", client.available_models)

# Send a message
try:
    response = client.generate(
        user_message="Hi, Good morning!",
        enableThinking=True
    )
    print("Model:", response["model"])
    print("Reasoning:", response["reasoning"])
    print("Reply:", response["reply"])

except GradientChatError as e:
    print("Request failed:", e)
```

## API Reference
`GradientChatClient`
```python
GradientChatClient(
    model="GPT OSS 120B",   # GPT OSS 120B (default) or Qwen3 235B
    cluster_mode="nvidia",  # nvidia (default) or hybrid, Qwen3 supports only hyrbid
    log_dir="logs",
    timeout=None            # default is 60 seconds
)
```
These parameters can also be set per request in the `generate` method.

`client.generate()`
```python
response = gradient_client.generate(
    user_message,           # required
    context_size=5,         # default is 15 and capped at a max of 50
    model="GPT OSS 120B",
    cluster_mode="nvidia",
    enableThinking=True,    # enables reasoning, False by default
    timeout=100,            # default timeout is 60 seconds
)
```


**OUTPUT:**
```python
{
    "reply": str,           # response to the user message
    "reasoning": str,       # reasoning used by the model
    "model": str            # model name
}
```

All parameters except `user_message` are optional. There is also a parameter called `conversation` of type `GradientConversation` which can be used to send custom conversation history as context.
```python
from gradient_chat import GradientConversation

custom_convo = GradientConversation(max_history=500)
custom_convo.add_user_message("Hi")
custom_convo.add_assistant_message("Hello!") # can also add reasoning text
custom_convo.add_user_message("How are you?")
custom_convo.add_assistant_message("I'm fine.")
client.generate("Hello", conversation=custom_convo)
```
Custom convo object can also be created directly as follows:
```python
custom_convo = [
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm fine.", "reasoningContent": "Responded with a polite, conventional reply to a common greeting to keep the conversation natural."},
    # add more
]
client.generate(user_message, custom_convo)
```

`client.get_model_info()`
*(can also use client.available_models)*
```python
['GPT OSS 120B', 'Qwen3 235B']
```

`client.get_conversation()`
```python
[
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there! How can I help you?", "reasoningContent": "Greeted the user."},
    {"role": "user", "content": "Can you tell me a joke?"},
    {"role": "assistant", "content": "Why don’t scientists trust atoms? Because they make up everything!"},
    {"role": "user", "content": "Thanks! What's the weather like today?"},
    {"role": "assistant", "content": "I cannot access real-time weather, but I recommend checking a local weather site.", "reasoningContent": "Explained limitations."}
]

```
## Error Handling
All fatal errors raise `GradientChatError`.  
Catch this single exception to handle any request failure.
```python
from gradient_chat import GradientChatClient, GradientChatError

client = GradientChatClient()

try:
    response = client.generate("Status update?")
    print(response)
except GradientChatError as e:
    msg = str(e)
    if "Timeout" in msg:
        print("Retrying with a higher timeout...")
        client.generate("Status update?", timeout=120)
    elif "Job Failed" in msg:
        print("Switching to another model...")
        client.generate("Status update?", model="Qwen3 235B")
    else:
        raise  # Let other errors (such as HTTP error, Network error) propagate
```
| Failure Type   | Cause                                               | Suggested Action                               |
| -------------- | --------------------------------------------------------- | ----------------------------------------------- |
| Request Timeout        | API took longer than timeout seconds.                     | Retry or increase timeout.                      |
| HTTP Error     | API returned non-2xx status.                               | Retry or investigate payload.                   |
| Network Error  | Connection issues, DNS failure, SSL errors, etc.          | Check connection / proxy.                       |
| Job Failed | API responded but never sent `status == "completed"`.     | Retry or switch to another model/cluster.       |

Non fatal errors are logged via `warnings.warn()` and do not stop execution.

## Disclaimer
This project is a personal undertaking and is not an official Gradient product. It is not affiliated with Gradient in any way and should not be mistaken as such.

## License
MIT License
