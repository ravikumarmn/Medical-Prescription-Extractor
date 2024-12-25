prompt_template_extract_ocr = """
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

Your primary goal is to accurately represent all the textual content from the image in a clear and structured Markdown format.

"""

prompt_template_extract_structured_data = """
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
    "Height": "",
    "Weight": "",
    "BMI": "",
    "SPO2": "",
    "HR": "",
    "B.P": "",
    "TEM": ""
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
