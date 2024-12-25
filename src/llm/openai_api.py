from src.utils import calculate_cost

from openai import OpenAI
from base_model import BaseModel

class OpenAIModel(BaseModel):
    def __init__(self, config) -> None:
        super().__init__(config)

    def configure(self, model_config):
        client = OpenAI(api_key=model_config["api_key"])
        return client

    def generate_content(self, input_data, generation_config):
        if "image" in input_data:
            response = self.model.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": input_data["prompt"]
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": input_data["image"]}
                        }
                    ]
                }],
                **generation_config
            )
        else:
            response = self.model.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": input_data["prompt"]}],
                **generation_config
            )
            
        usage = response.usage
        cost_in_dollers = calculate_cost(
            usage.prompt_tokens,
            usage.completion_tokens,
        )
        return response.choices[0].message.content, cost_in_dollers

# import os
# from openai import OpenAI
# from src.llm.gemini_api import GeminiModel
# from src import extractor
# from src.utils import calculate_cost
# from langchain_core.output_parsers import JsonOutputParser
# import json
# import google.generativeai as genai

# class OpenAIModel(GeminiModel):
#     def __init__(
#         self,
#         model_name: str = "gpt-4o-mini",
#         api_key: str = None,
#         api_key_env_var: str = "OPENAI_API_KEY",
#     ):
#         super().__init__(model_name, api_key, api_key_env_var)

#     def _configure(self, config):
#         import openai  # Import OpenAI library

#         openai.api_key = self.api_key
#         return openai.chat.completions.create

#     def generate_json_content(
#         self, content: str | list, base64_image: str, max_tokens: int = 500
#     ):
#         response = OpenAI.chat.completions.create(
#             model=self.model_name,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "text",
#                             "text": (
#                                 content
#                                 if isinstance(content, str)
#                                 else " ".join(content)
#                             ),
#                         },
#                         {"type": "image_url", "image_url": {"url": f"{base64_image}"}},
#                     ],
#                 }
#             ],
#             max_tokens=max_tokens,
#         )

#         usage_metadata = response["usage"]
#         cost_in_dollars = calculate_cost(
#             usage_metadata["prompt_tokens"], usage_metadata["completion_tokens"]
#         )
#         response_text = response["choices"][0]["message"]["content"]

#         try:
#             parser = JsonOutputParser()
#             result = parser.parse(response_text)
#         except json.JSONDecodeError as e:
#             print("Failed to parse JSON:", e)
#             result = None

#         return result, cost_in_dollars


# # Usage
# if __name__ == "__main__":
#     model = OpenAIModel()
#     print()


# api_key = os.getenv('OPENAI_API_KEY', None)
#             if not api_key:
#                 return jsonify({'error': 'API key is missing.'}), 500

#             client = OpenAI(api_key=api_key)

# response = client.chat.completions.create(
#                 model=model,
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": [
#                             {
#                                 "type": "text",
#                                 "text": prompt
#                             },
#                             {
#                                 "type": "image_url",
#                                 "image_url": {"url": f"{base64_image}"}
#                             }
#                         ]
#                     }
#                 ],
#                 max_tokens=max_tokens
#             )


# def process_response(response, image_path):
#     json_string = response.choices[0].message.content
#     json_string = json_string.replace("```json\n", "").replace("\n```", "")
#     json_data = json.loads(json_string)
#     filename_without_extension = os.path.splitext(os.path.basename(image_path))[0]
#     json_filename = f"{filename_without_extension}.json"
#     with open(f"./results/{json_filename}", "w") as file:
#         json.dump(json_data, file, indent=4)
#     print(f"JSON data saved to {json_filename}")
#     return json_data
