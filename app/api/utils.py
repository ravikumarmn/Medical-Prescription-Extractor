import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load configuration
config_path = Path(__file__).parent.parent.parent / "config.json"
with open(config_path) as f:
    CONFIG = json.load(f)

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in environment variables")

genai.configure(api_key=GOOGLE_API_KEY)

class ProcessingError(Exception):
    """Custom exception for processing errors."""
    pass

def get_prompt_template_extract_ocr() -> str:
    """Returns the prompt template for OCR extraction."""
    return """
    You are tasked with extracting all textual information from the provided image accurately, even if the image is blurry or contains glare. Extract both plain text and structured tabular information.

    ### Guidelines for Extraction:

    1. **General Text Extraction**:
    - Extract all visible text, including titles, headers, annotations, and any other written content.
    - Preserve:
        - The original case (upper/lowercase letters).
        - All punctuation marks and formatting.
    - Ensure all extracted text is direct and accurate—no interpretations or omissions.

    2. **Table and Structured Content Extraction**:
    - Identify structured tabular data, prescriptions, or organized lists within the image.
    - Represent these in Markdown format using `|` for columns and `-` for headers.
    - Maintain the integrity of the content: include all rows, columns, and line breaks using `<br>` tags if needed.

    3. **Content Organization**:
    - Deliver extracted text in Markdown format.
    - Maintain a logical flow, similar to how the content is presented in the image.
    - Do not interpret or restructure the text beyond what is explicitly visible.
    """

def get_prompt_template_extract_structured_data() -> str:
    """Returns the prompt template for structured data extraction."""
    return """
    I have the following raw OCR-extracted text from a medical prescription. Your task is to transform this raw text into a structured JSON format as specified below.

    **Requirements:**

    1. **Patient Information**:
    - Extract the following details:
        - **Name**: The patient's full name.
        - **Age**: Patient's age.
        - **Gender**: Patient's gender.
        - **Date**: Date of the prescription.
        - **Height, Weight, BMI, SPO2, HR, B.P, TEM**: Extract these metrics if available.

    2. **Doctor Information**:
    - **Name**: The name of the doctor.
    - **Graduation**: Doctor's qualifications (e.g., MD, DM Cardiology).
    - **Hospital Name**: The name of the hospital or clinic.
    - **Location**: Address or contact information of the hospital.
    - **License Number**: The doctor's license/registration number.

    3. **Medication Information**:
    - Extract the list of medicines prescribed. For each medicine, include:
        - **Medicine Name**: The name of the medicine.
        - **Dosage**: The prescribed dosage (e.g., 100 mg, 5 ml).
        - **Form**: Type of medicine (e.g., tablet, liquid).
        - **Quantity**: Amount prescribed (e.g., sheets, bottles).
        - **When to Use**: Times of day to take the medicine based on the dosage pattern.

    4. **Symptoms**:
    - Extract the list of symptoms or reasons for prescription mentioned by the doctor.

    5. **Doctor's Note**:
    - Include any additional remarks or notes provided by the doctor.

    Raw OCR Text:
    {{raw_ocr_text}}
    """

def calculate_cost(prompt_token_count: int, completion_token_count: int, model_name: str = CONFIG["model"]["name"]) -> float:
    """Calculate API usage cost based on token counts."""
    per_million_usd = {
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-flash-001": {"input": 0.075, "output": 0.30},
        "gemini-1.5-flash-002": {"input": 0.075, "output": 0.30},
    }
    
    pricing = per_million_usd.get(model_name)
    if not pricing:
        raise ValueError(f"Unsupported model: {model_name}")

    input_cost = (max(0, prompt_token_count) / 1_000_000) * pricing["input"]
    output_cost = (max(0, completion_token_count) / 1_000_000) * pricing["output"]
    
    return round(input_cost + output_cost, 6)

def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess the image for better OCR results."""
    try:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        denoised = cv2.fastNlMeansDenoising(binary)
        
        h, w = denoised.shape
        max_size = (CONFIG["image_processing"]["max_width"], CONFIG["image_processing"]["max_height"])
        if h > max_size[1] or w > max_size[0]:
            ratio = min(max_size[0]/w, max_size[1]/h)
            new_size = (int(w*ratio), int(h*ratio))
            denoised = cv2.resize(denoised, new_size, interpolation=cv2.INTER_AREA)
        
        return Image.fromarray(denoised)
    except Exception as e:
        raise ProcessingError(f"Error preprocessing image: {str(e)}")

def process_image_ocr(image: Image.Image, prompt: str) -> str:
    """Process image through Gemini Vision API for OCR."""
    try:
        model = genai.GenerativeModel(CONFIG["model"]["name"])
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        raise ProcessingError(f"Error processing image with Gemini Vision API: {str(e)}")

def process_structured_data(ocr_text: str, prompt: str) -> Dict[str, Any]:
    """Process OCR text to extract structured data."""
    try:
        model = genai.GenerativeModel(CONFIG["model"]["name"])
        prompt_with_text = prompt.replace("{{raw_ocr_text}}", ocr_text)
        response = model.generate_content(prompt_with_text)
        
        # Extract JSON from the response
        text = response.text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            
        return json.loads(text)
    except json.JSONDecodeError:
        # Return a structured error response
        return {
            "Patient Information": {},
            "Doctor Information": {},
            "Medication Information": [],
            "Symptoms": [],
            "Doctor Note": "Error: Could not parse the response"
        }
    except Exception as e:
        raise ProcessingError(f"Error extracting structured data: {str(e)}")

def process_prescription(image: Image.Image) -> Dict[str, Any]:
    """Process prescription image and return both OCR and structured results."""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Get OCR results
        ocr_prompt = get_prompt_template_extract_ocr()
        ocr_text = process_image_ocr(processed_image, ocr_prompt)
        
        # Get structured data
        structured_prompt = get_prompt_template_extract_structured_data()
        structured_data = process_structured_data(ocr_text, structured_prompt)
        
        return {
            # "ocr_text": ocr_text, # Uncomment this line if you want to return the OCR text
            "structured_data": structured_data
        }
    except Exception as e:
        raise ProcessingError(f"Error processing prescription: {str(e)}")
