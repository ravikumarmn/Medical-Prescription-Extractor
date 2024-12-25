import os
import json
import time
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

import streamlit as st
from PIL import Image
import google.generativeai as genai
from langchain_core.output_parsers import JsonOutputParser
import numpy as np
import cv2
from dotenv import load_dotenv
import pdf2image

# Load environment variables
load_dotenv()

# Load configuration
config_path = Path(__file__).parent.parent / "config.json"
with open(config_path) as f:
    CONFIG = json.load(f)

# Configure Gemini API
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in streamlit secrets")

genai.configure(api_key=GOOGLE_API_KEY)

class ProcessingError(Exception):
    """Custom exception for processing errors."""
    pass

@st.cache_data
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

    ### Specific Requirements:
    - Extract the following details in their entirety if visible:
    - Patient Information, including name, age, gender, date, height, weight, BMI, SPO2, HR, B.P, TEM.
    - Hospital Information, including hospital name, location, doctor details (name, specialization, license number, and contact).
    - Medication Information, including medicine name, dosage, form (e.g., tablet, liquid), and quantity (e.g., sheets, bottles).
    - Doctor's notes or any other remarks.

    ### Output Format:
    - Present all extracted data in Markdown format without rephrasing.
    - Ensure completeness and high fidelity of the output by avoiding omission or assumption.
    - Avoid unnecessary annotations or commentary, providing the content in a raw form.
    """

@st.cache_data
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
        - **When to Use**: Times of day to take the medicine based on the dosage pattern (e.g., ["morning", "afternoon", "evening"]).

    4. **Symptoms**:
    - Extract the list of symptoms or reasons for prescription mentioned by the doctor (e.g., hypothyroidism, neuropathy).

    5. **Doctor's Note**:
    - Include any additional remarks or notes provided by the doctor.

    **Output Format**:
    ```json
    {
    "Patient Information": {
        "Name": "",
        "Age": "",
        "Gender": "",
        "Date": "",
    },
    "Doctor Information": {
        "Name": "",
        "Graduation": "",
        "Hospital Name": "",
        "Location": "",
        "License Number": ""
    },
    "Medication Information": [
        {
        "Medicine Name": "",
        "Dosage": "",
        "Form": "",
        "Quantity": ""
        }
    ],
    "Symptoms": [],
    "Doctor Note": ""
    }

    Raw OCR Text:
    {{raw_ocr_text}}

    **Instructions**:
    - Extract and populate each field based on the raw text.
    - Maintain the original wording and data accuracy; avoid interpretations.
    - If a detail is missing or unclear, leave the field empty or use "Not Available."
    """

def calculate_cost(prompt_token_count: int, completion_token_count: int, model_name: str = CONFIG["model"]["name"]) -> float:
    """
    Calculate API usage cost based on token counts.
    
    Args:
        prompt_token_count: Number of input tokens
        completion_token_count: Number of output tokens
        model_name: Name of the model being used
    
    Returns:
        float: Total cost in USD
    """
    per_million_usd = {
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-flash-001": {"input": 0.075, "output": 0.30},
        "gemini-1.5-flash-002": {"input": 0.075, "output": 0.30},
    }
    
    pricing = per_million_usd.get(model_name)
    if not pricing:
        raise ValueError(f"Unsupported model: {model_name}")

    # Calculate costs per million tokens
    input_cost = (max(0, prompt_token_count) / 1_000_000) * pricing["input"]
    output_cost = (max(0, completion_token_count) / 1_000_000) * pricing["output"]
    
    return round(input_cost + output_cost, 6)  # Round to 6 decimal places for USD

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
        raise ProcessingError(f"Image preprocessing failed: {str(e)}")

async def process_image_ocr(image: Image.Image, prompt: str) -> Dict[str, Any]:
    """Process image through Gemini Vision API for OCR."""
    try:
        start_time = time.perf_counter()  # Use perf_counter for higher precision
        model = genai.GenerativeModel(CONFIG["model"]["name"])
        response = await model.generate_content_async([prompt, image])
        end_time = time.perf_counter()
        
        if not response.text:
            raise ProcessingError("No text extracted from image")
        
        # Get token counts from response
        token_count = getattr(response.candidates[0], 'token_count', 0) if hasattr(response, 'candidates') else 0
        
        return {
            "text": response.text,
            "time_taken": round(end_time - start_time, 3),  # Round to milliseconds
            "prompt_tokens": token_count,
            "completion_tokens": len(response.text.split()) * 2  # Approximate completion tokens
        }
    except Exception as e:
        raise ProcessingError(f"OCR processing failed: {str(e)}")

async def process_structured_data(ocr_text: str, prompt: str) -> Dict[str, Any]:
    """Process OCR text to extract structured data."""
    try:
        start_time = time.perf_counter()
        model = genai.GenerativeModel(CONFIG["model"]["name"])
        parser = JsonOutputParser()
        
        full_prompt = prompt + "\n\nText:\n" + ocr_text
        response = await model.generate_content_async(full_prompt)
        parsed_data = parser.parse(response.text)
        end_time = time.perf_counter()
        
        if not parsed_data:
            raise ProcessingError("Failed to parse structured data")
        
        # Get token counts
        token_count = getattr(response.candidates[0], 'token_count', 0) if hasattr(response, 'candidates') else 0
        
        return {
            "data": parsed_data,
            "time_taken": round(end_time - start_time, 3),
            "prompt_tokens": token_count,
            "completion_tokens": len(str(parsed_data).split()) * 2  # Approximate completion tokens
        }
    except Exception as e:
        raise ProcessingError(f"Structured data extraction failed: {str(e)}")

async def process_prescription(image: Image.Image, ocr_prompt: str, structured_prompt: str) -> Dict[str, Any]:
    """Process prescription image and return both OCR and structured results."""
    try:
        ocr_result = await process_image_ocr(image, ocr_prompt)
        structured_result = await process_structured_data(ocr_result["text"], structured_prompt)
        
        return {
            "success": True,
            "ocr_result": ocr_result,
            "structured_result": structured_result
        }
    except ProcessingError as e:
        st.error(str(e))
        return {"success": False, "error": str(e)}

def main():
    """Main application function."""
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")
    
    # Create three columns with the middle one being wider
    left_col, center_col, right_col = st.columns([1, 2, 1])
    
    with st.sidebar:
        st.title("Upload Prescription")
        uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'jpeg', 'pdf'])
        
        if uploaded_file is not None:
            # Handle PDF files
            if uploaded_file.type == "application/pdf":
                # Convert first page of PDF to image
                pdf_pages = pdf2image.convert_from_bytes(uploaded_file.read())
                if not pdf_pages:
                    st.error("Could not process PDF file. Please ensure it's not empty.")
                    return
                image = pdf_pages[0]
                # Display the converted PDF
                st.image(image, caption="Uploaded PDF (First Page)")
            else:
                # Handle image files
                image = Image.open(uploaded_file)
                # Display the uploaded image
                st.image(image, caption="Uploaded Image")
    
    with center_col:
        st.title("Medical Prescription Text Extraction")
        
        if uploaded_file is not None:
            try:
                with st.spinner("Processing..."):
                    if uploaded_file.type == "application/pdf":
                        # We already have the image from PDF conversion above
                        processed_image = preprocess_image(image)
                    else:
                        # Process the uploaded image
                        image = Image.open(uploaded_file)
                        processed_image = preprocess_image(image)
                    
                    results = asyncio.run(process_prescription(
                        processed_image,
                        get_prompt_template_extract_ocr(),
                        get_prompt_template_extract_structured_data()
                    ))
                    
                    if not results["success"]:
                        return
                    
                    if results["structured_result"]["data"]:
                        st.subheader("Structured Information")
                        st.json(results["structured_result"]["data"])
                    else:
                        st.warning("Could not extract structured information from the text")
                    
                    # Calculate costs with improved token counting
                    ocr_cost = calculate_cost(
                        results["ocr_result"]["prompt_tokens"],
                        results["ocr_result"]["completion_tokens"]
                    )
                    
                    structured_cost = calculate_cost(
                        results["structured_result"]["prompt_tokens"],
                        results["structured_result"]["completion_tokens"]
                    )
                    
                    total_time = round(
                        results["ocr_result"]["time_taken"] + 
                        results["structured_result"]["time_taken"],
                        3
                    )
                    
                    metadata = {
                        "Processing Time": {
                            "OCR": f"{results['ocr_result']['time_taken']:.3f}s",
                            "Formatting": f"{results['structured_result']['time_taken']:.3f}s",
                            "Total": f"{total_time:.3f}s"
                        },
                        "Cost Information": {
                            "OCR": f"${ocr_cost:.6f}",
                            "Formatting": f"${structured_cost:.6f}",
                            "Total": f"${(ocr_cost + structured_cost):.6f}"
                        }
                    }
                    
                    st.subheader("Metadata")
                    st.json(metadata)
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
