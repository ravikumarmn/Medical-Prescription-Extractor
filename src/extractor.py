import time

from src.prompts import (
    prompt_template_extract_ocr,
    prompt_template_extract_structured_data,
)
import os

# gemini_model = GeminiModel("gemini-1.5-flash")


from llm.gemini_api import GeminiModel
from llm.openai_api import OpenAIModel
from src.utils import encode_image

class TextExtractor:
    def __init__(self, config):
        self.config = config

    def process_image(self, image_path, prompt, model_choice="gemini"):
        if not image_path:
            model_input = {"prompt": prompt}
        else:
            if not os.path.exists(image_path):
                raise ValueError("Image path is incorrect or file does not exist")
            
            if model_choice == "gemini":
                model = GeminiModel(self.config)
                with open(image_path, 'rb') as img_file:
                    image_data = img_file.read()
                model_input = {
                    "prompt": prompt,
                    "image": image_data
                }
                model_config = {
                    "temperature": 0.7,
                    "max_output_tokens": 500,
                }
            else:
                model = OpenAIModel(self.config)
                base64_image = f"data:image/png;base64,{encode_image(image_path)}"
                model_input = {
                    "prompt": prompt,
                    "image": base64_image
                }
                model_config = {
                    "temperature": 0.7,
                    "max_tokens": 500
                }

        response, cost = model.generate_content(model_input, model_config)
        return {
            "model_used": model_choice,
            "response": response,
            "cost": cost
        }
    

if __name__ == "__main__":
    config = {
        "gemini": {
            "api_key": os.getenv("GOOGLE_API_KEY"),
            "model_name": "gemini-1.5-pro-vision"
        },
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model_name": "gpt-4-vision-preview"
        }
    }
    
    image_path = "path/to/your/image.jpg"
    prompt = "Describe what you see in this image"
    
    processor = TextExtractor(config)
    
    # Test with Gemini
    gemini_results = processor.process_image(image_path, prompt, "gemini")
    print(f"\nGemini Response: {gemini_results['response']}")
    print(f"Gemini Cost: ${gemini_results['cost']}")
    
    # Test with OpenAI
    openai_results = processor.process_image(image_path, prompt, "openai") 
    print(f"\nOpenAI Response: {openai_results['response']}")
    print(f"OpenAI Cost: ${openai_results['cost']}")


# class TextExtractor:
#     def __init__(self, model_name="gpt-4o"):
#         self.model_name = model_name
#         self.model = self._initialize_model()

#     def _initialize_model(self):
#         if "gemini" in self.model_name:
#             from src.llm.gemini_api import GeminiModel
#             return GeminiModel(self.model_name)
#         elif "gpt" in self.model_name:
#             from src.llm.openai_api import OpenAIModel
#             return OpenAIModel(self.model_name)
#         else:
#             raise NotImplementedError

#     def structured_ocr(self, image, context_txt="Here is the medical prescription:"):
#         start_time = time.time()  # Start the timer for OCR
#         response_text, cost_in_dollars = self.model.generate_content(
#             [prompt_template_extract_ocr + "\n\n" + context_txt, image]
#         )
#         end_time = time.time()  # End the timer for OCR
#         ocr_time_taken = end_time - start_time  # Calculate OCR time taken

#         if response_text:
#             return {
#                 "cost": cost_in_dollars,
#                 "response": response_text,
#                 "time_taken": ocr_time_taken,
#             }

#     def extract_required_data(self, raw_ocr_text, base64_image=None):
#         if "gpt" in self.model_name and base64_image is not None:
#             response, cost_in_dollars = self.model.generate_json_content(
#                 raw_ocr_text, base64_image
#             )
#         elif "gemini" in self.model_name:
#             start_time = time.time()  # Start the timer for structured data extraction
#             prompt = prompt_template_extract_structured_data.replace(
#                 "{{raw_ocr_text}}", raw_ocr_text
#             )
#             response, cost_in_dollars = self.model.generate_json_content(prompt)
#             end_time = time.time()  # End the timer for structured data extraction
#             structured_data_time_taken = (
#                 end_time - start_time
#             )  # Calculate time taken for formatting

#             return {
#                 "cost": cost_in_dollars,
#                 "response": response,
#                 "time_taken": structured_data_time_taken,
#             }


# # Usage
# if __name__ == "__main__":
#     import PIL.Image

#     uploaded_file = PIL.Image.open("data/original_temp_image.jpeg")
#     extractor = TextExtractor(model_name="gpt-4o-mini")
#     result = extractor.structured_ocr(
#         image=uploaded_file, context_txt="Here is the medical prescription:"
#     )
#     print()


# import os
# from openai import OpenAI
# from src.llm.gemini_api import GeminiModel
# from src import extractor
# from src.utils import calculate_cost
# from langchain_core.output_parsers import JsonOutputParser
# import json
# import google.generativeai as genai
# from abc import ABC, abstractmethod


# class BaseModel:
#     def __init__(self, config: dict) -> None:
#         self.model = self.configure(config["model_config"])
#         self.model_name = self.model["model_name"]

#     @abstractmethod
#     def configure(self, model_config: dict):
#         raise NotImplementedError

#     @abstractmethod
#     def generate_content(self, input_data: dict, generation_config: dict):
#         raise NotImplementedError

#     def generate_json_content(self, input_data, generation_config):
#         response_text, cost_in_dollers = self.generate_content(
#             self, input_data, generation_config
#         )
#         try:
#             parser = JsonOutputParser()
#             result = parser.parse(response_text)
#         except json.JSONDecodeError as e:
#             print("Failed to parse JSON:", e)
#             result = None
#         return result, cost_in_dollers

#     def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
#         cost_in_dollers = calculate_cost(
#             prompt_tokens,
#             completion_tokens,
#             model_name=self.model_name,
#         )
#         return 0.0 if cost_in_dollers is None else cost_in_dollers


# class GeminiModel(BaseModel):
#     def __init__(self, config) -> None:
#         super().__init__(config)

#     def configure(self, model_config):
#         genai.configure(api_key=model_config["self.api_key"])
#         return genai.GenerativeModel(
#             model_name=model_config["model_name"],
#         )

#     def generate_content(self, input_data, generation_config):
#         if "image" in input_data:
#             response = self.model.generate_content(
#                 contents=[input_data["prompt"], input_data["image"]], 
#                 generation_config=generation_config
#             )
#         else:
#             response = self.model.generate_content(
#                 contents=input_data["prompt"],
#                 generation_config=generation_config
#             )
            
#         usage_metadata = response.usage_metadata
#         cost_in_dollers = self.calculate_cost(
#             usage_metadata.prompt_token_count,
#             usage_metadata.candidates_token_count
#         )
#         return response.text, cost_in_dollers


# from openai import OpenAI

# class OpenAIModel(BaseModel):
#     def __init__(self, config) -> None:
#         super().__init__(config)

#     def configure(self, model_config):
#         client = OpenAI(api_key=model_config["api_key"])
#         return client

#     def generate_content(self, input_data, generation_config):
#         if "image" in input_data:
#             response = self.model.chat.completions.create(
#                 model=self.model_name,
#                 messages=[{
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "text",
#                             "text": input_data["prompt"]
#                         },
#                         {
#                             "type": "image_url",
#                             "image_url": {"url": input_data["image"]}
#                         }
#                     ]
#                 }],
#                 **generation_config
#             )
#         else:
#             response = self.model.chat.completions.create(
#                 model=self.model_name,
#                 messages=[{"role": "user", "content": input_data["prompt"]}],
#                 **generation_config
#             )
            
#         usage = response.usage
#         cost_in_dollers = calculate_cost(
#             usage.prompt_tokens,
#             usage.completion_tokens,
#         )
#         return response.choices[0].message.content, cost_in_dollers



# def process_image_with_model(image_path, prompt, config, model_choice="gemini"):

#     if model_choice not in ["gemini", "openai"]:
#         raise ValueError("model_choice must be either 'gemini' or 'openai'")

#     if not image_path:
#         model_input = {
#             "prompt": prompt
#         }
#     else:
#         if not os.path.exists(image_path):
#             raise ValueError("Image path is incorrect or file does not exist")

#         if model_choice == "gemini":
#             model = GeminiModel(config)
#             # Prepare image data for Gemini
#             with open(image_path, 'rb') as img_file:
#                 image_data = img_file.read()
            
#             model_input = {
#                 "prompt": prompt,
#                 "image": image_data
#             }
#             model_config = {
#                 "temperature": 0.7,
#                 "max_output_tokens": 500,
#             }

#         else:  # OpenAI
#             model = OpenAIModel(config)
#             # Base64 encode image for OpenAI
#             base64_image = f"data:image/png;base64,{encode_image(image_path)}"
            
#             model_input = {
#                 "prompt": prompt,
#                 "image": base64_image
#             }
#             model_config = {
#                 "temperature": 0.7,
#                 "max_tokens": 500
#             }

#     response, cost = model.generate_content(model_input, model_config)

#     return {
#         "model_used": model_choice,
#         "response": response,
#         "cost": cost
#     }

# if __name__ == "__main__":
#     config = {
#         "gemini": {
#             "api_key": os.getenv("GOOGLE_API_KEY"),
#             "model_name": "gemini-1.5-pro-vision"
#         },
#         "openai": {
#             "api_key": os.getenv("OPENAI_API_KEY"),
#             "model_name": "gpt-4-vision-preview"
#         }
#     }
    
#     image_path = "path/to/your/image.jpg"
#     prompt = "Describe what you see in this image"
    
#     # Use Gemini
#     gemini_results = process_image_with_model(image_path, prompt, config, "gemini")
#     print(f"\nGemini Response: {gemini_results['response']}")
#     print(f"Gemini Cost: ${gemini_results['cost']}")
    
#     # Use OpenAI
#     openai_results = process_image_with_model(image_path, prompt, config, "openai")
#     print(f"\nOpenAI Response: {openai_results['response']}")
#     print(f"OpenAI Cost: ${openai_results['cost']}")



