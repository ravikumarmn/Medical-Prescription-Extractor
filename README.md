**https://www.vishrx.com**

# OCR Prescription Extractor

An AI-powered application that extracts and structures medical prescription information using OCR and Gemini Vision AI.

## Features

- Canvas-based prescription input
- AI-powered text extraction and structuring
- Real-time processing with cost and time tracking
- Clean and intuitive user interface

## Prerequisites

- Python 3.10 or higher
- Gemini API key ([Get it here](https://makersuite.google.com/app/apikey))
- pip (Python package manager)

## Installation

1. **Set Up Python Environment**
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Project in Development Mode**
   ```bash
   pip install -e .
   ```

## Configuration
### Configuration

1. **Set up API Key**
   You need to configure your Gemini API key in two places. If the files don't exist, create them:
   ```bash
   mkdir -p .streamlit
   touch .streamlit/secrets.toml
   touch .env
   ```

   a. Add your API key to `.streamlit/secrets.toml`:
   ```toml
   GOOGLE_API_KEY = "your_gemini_api_key"
   ```

   b. Add your API key to `.env`:
   ```toml
   GOOGLE_API_KEY="your_gemini_api_key"
   ```

## Running the Application

1. **Start the Streamlit Server**
   ```bash
   streamlit run app/streamlit_run.py
   ```

2. **Access the Application**
   - The application will automatically open in your default web browser
   - If not, visit: http://localhost:8501


# API  

* `http://localhost:8000/api/health`
    *   Just to very the endopoint is working.
    *   It returns a 200 status code.
    
* `http://localhost:8000/api/process`
    *   This is the main endpoint for processing the image.
    *   It accepts a POST request with the image file as the payload.

## Usage

1. Write or draw your prescription on the canvas
2. Click "Extract Text" to process the image
3. View the structured information and processing metadata

## Troubleshooting

If you encounter any issues:

1. **API Key Errors**
   - Verify your API key is correctly set in both `.env` and `.streamlit/secrets.toml`
   - Check that the API key is valid and has necessary permissions
   - Make sure there are no extra spaces or quotes around the API key

2. **Installation Issues**
   - Ensure you're using Python 3.8 or higher: `python --version`
   - Try reinstalling dependencies: `pip install -r requirements.txt --force-reinstall`
   - Make sure your virtual environment is activated

3. **Application Errors**
   - Check the console for error messages
   - Verify all configuration files are properly formatted
   - Ensure you have an active internet connection
   - Check if port 8501 is available (default Streamlit port)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
