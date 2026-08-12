import os
from litellm import completion
from litellm.exceptions import APIError

class LLMManager:
    """Manages chat completion interactions with foundational language models."""
    def __init__(self, model_name: str = "gemini-3.6-flash"):
        self.model_name = model_name

    def generate_chat_response(self, messages: list[dict]) -> str:
        """Executes a chat completion call using an input history list payload."""
        try:
            response = completion(
                model=self.model_name,
                messages=messages,
                custom_llm_provider="gemini",
                api_key=os.getenv("GEMINI_API_KEY")
            )
            return response.choices[0].message.content
        except APIError as api_err:
            return f"❌ Upstream AI Provider Connection Error: {str(api_err)}"
        except Exception as e:
            return f"⚠️ An unexpected calculation error occurred: {str(e)}"
