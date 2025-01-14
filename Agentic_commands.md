# Agent API Example

##  Key sk-pro
```
j-rdp2MQmUuoeT5wob1h0MuifPsnC3MA8BfQPUNmQp39UVitPvI0jN_a2y8hF1YyZp0FmwgRe148T3BlbkFJ2V4qovJ6r0JRJv7MDSPUOsjW2_aLI4okmhOi5fXF2_w_5HfH7arqsYPK_Mqt1mq7k_umkkuQ0A
```

## Install OpenAI Python Package
```bash
!pip install openai
```

## Python Code
```python
import openai

def chat_with_gpt4(api_key, prompt, model="gpt-4", max_tokens=10000):
    """
    Makes a call to OpenAI's ChatGPT API (GPT-4).

    Args:
        api_key (str): Your OpenAI API key.
        prompt (str): The prompt to send to the model.
        model (str): The model to use (default is gpt-4).
        max_tokens (int): The maximum number of tokens to generate.

    Returns:
        str: The response from the model.
    """
    openai.api_key = api_key

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )

        return response["choices"][0]["message"]["content"]

    except openai.error.OpenAIError as e:
        return f"An error occurred: {str(e)}"

# Example usage
if __name__ == "__main__":
    api_key = "your_openai_api_key_here"  # Replace with your actual OpenAI API key
    prompt = "Hello, ChatGPT! Can you explain how neural networks work?"

    response = chat_with_gpt4(api_key, prompt)
    print("Response from ChatGPT:")
    print(response)
