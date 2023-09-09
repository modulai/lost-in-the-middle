import os

import openai
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_openai_chat_completion(
    model: str, temperature: float, top_p: float, max_tokens: int, system_message: str, user_message: str
) -> str:
    def create_chat_completion():
        return openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    try:
        response = create_chat_completion()
    except Exception as e:
        print(f"Error: {e}")
        print("Let's sleep for 40s")
        time.sleep(40)
        response = create_chat_completion()

    return response["choices"][0]["message"]["content"]
