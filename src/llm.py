import json
import os

from openai import OpenAI


class LLM:
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI()

    def generate(self, prompt: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=1000,  # https://github.com/tsinghua-fib-lab/AgentMove/blob/main/models/llm_api.py#L137
        )
        result = response.choices[0].message.content
        return json.loads(result)


class Gemini(LLM):
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
