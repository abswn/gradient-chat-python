from gradient_chat import GradientChatClient

if __name__ == "__main__":
    # Create client
    gradient_client = GradientChatClient(log_dir="example_logs")

    # Show available models
    print("Available Models:", gradient_client.available_models)

    # User message
    user_message = "Hi, Goodmorning!"
    
    # Generate response (enableThinking per request)
    response = gradient_client.generate(
        user_message, 
        context_size=5, 
        enableThinking=True
    )

    # Print question and response
    print("Question:", user_message)
    print("Model:", response["model"])
    print("Reasoning:", response["reasoning"])
    print("Reply:", response["reply"])
