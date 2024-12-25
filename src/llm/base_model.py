from abc import ABC, abstractmethod
from langchain_core.output_parsers import JsonOutputParser
from src.utils import calculate_cost
import json

class BaseModel:
    def __init__(self, config: dict) -> None:
        self.model = self.configure(config["model_config"])
        self.model_name = self.model["model_name"]

    @abstractmethod
    def configure(self, model_config: dict):
        raise NotImplementedError

    @abstractmethod
    def generate_content(self, input_data: dict, generation_config: dict):
        raise NotImplementedError

    def generate_json_content(self, input_data, generation_config):
        response_text, cost_in_dollers = self.generate_content(
            self, input_data, generation_config
        )
        try:
            parser = JsonOutputParser()
            result = parser.parse(response_text)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            result = None
        return result, cost_in_dollers

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost_in_dollers = calculate_cost(
            prompt_tokens,
            completion_tokens,
            model_name=self.model_name,
        )
        return 0.0 if cost_in_dollers is None else cost_in_dollers