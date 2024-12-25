# import os
import google.generativeai as genai
from src.utils import calculate_cost

# from langchain_core.output_parsers import JsonOutputParser
# import json
from base_model import BaseModel

class GeminiModel(BaseModel):
    def __init__(self, config) -> None:
        super().__init__(config)

    def configure(self, model_config):
        genai.configure(api_key=model_config["self.api_key"])
        return genai.GenerativeModel(
            model_name=model_config["model_name"],
        )

    def generate_content(self, input_data, generation_config):
        if "image" in input_data:
            response = self.model.generate_content(
                contents=[input_data["prompt"], input_data["image"]], 
                generation_config=generation_config
            )
        else:
            response = self.model.generate_content(
                contents=input_data["prompt"],
                generation_config=generation_config
            )
            
        usage_metadata = response.usage_metadata
        cost_in_dollers = self.calculate_cost(
            usage_metadata.prompt_token_count,
            usage_metadata.candidates_token_count
        )
        return response.text, cost_in_dollers

# class GeminiModel:
#     def __init__(
#         self,
#         model_name: str = "gemini-1.5-flash",
#         api_key: str = None,
#         api_key_env_var: str = "GEMINI_API_KEY",
#     ):
#         self.api_key = api_key or os.environ[api_key_env_var]
#         if not self.api_key:
#             raise ValueError("API key is missing. Please provide a valid API key.")

#         self.model = self._configure()

#     def _configure(self, config):
#         genai.configure(api_key=self.api_key)
#         default_params = {
#             "candidate_count": 1,
#             "temperature": 1.0,
#             "top_p": 0.95,
#             "top_k": 20,
#             # "max_temperature": 2.0, #TODO: Answer why max_temperature parameter is not taking.
#         }
#         default_params.update(config["generation_params"])
#         generation_config = genai.GenerationConfig(**default_params)
#         return genai.GenerativeModel(
#             model_name=config["model_name"],
#             generation_config=generation_config,
#         )

#     def generate_content(self, content: str | list, generation_config=None):
#         response = self.model.generate_content(
#             contents=content, generation_config=generation_config
#         )
#         usage_metadata = response.usage_metadata
#         cost_in_dollers = calculate_cost(
#             usage_metadata.prompt_token_count, usage_metadata.candidates_token_count
#         )
#         response_text = response.text
#         return response_text, cost_in_dollers

#     def generate_json_content(self, content: str | list, generation_config=None):

#         response = self.model.generate_content(
#             contents=content,
#             generation_config=generation_config,
#         )
#         usage_metadata = response.usage_metadata
#         cost_in_dollers = calculate_cost(
#             usage_metadata.prompt_token_count, usage_metadata.candidates_token_count
#         )
#         # response_text = response.text

#         try:
#             parser = JsonOutputParser()
#             result = parser.parse(response.text)
#         except json.JSONDecodeError as e:
#             print("Failed to parse JSON:", e)
#             result = None

#         return result, cost_in_dollers


# # Usage Example
# if __name__ == "__main__":
#     try:
#         gemini_model = GeminiModel()
#         model = gemini_model.get_model(model_name="gemini-1.5-flash", temperature=1.5)
#         print()
#     except ValueError as e:
#         print(f"Error: {e}")
