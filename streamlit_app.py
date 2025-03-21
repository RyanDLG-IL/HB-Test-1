import os
import csv
import re
import io
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env (ensure GEMINI_API_KEY is set)
load_dotenv()

def parse_analysis_output(output_text: str) -> dict:
    """
    Parse the model's output text into a structured dictionary.

    Expected sub-headings in the output text:
      - Question Analysis Summary:
          * Required knowledge and skills:
          * Key vocabulary terms:
          * Common misconceptions or challenges:
      - Vocabulary Analysis:
          * Vocabulary Terms:
          * Definitions:
          * Special Attention:
      - Implementation Recommendations:
          * Teaching Approaches:
          * Scaffolding:
          * Sequencing:

    Returns:
        A dictionary with keys:
          "QA_Knowledge-and-skills", "QA_Key-Vocabulary", "QA_Common-misconceptions",
          "VA_Vocabulary-Terms", "VA_Definitions", "VA_Special-Attention",
          "IR_Teaching-Approaches", "IR_Scaffolding", "IR_Sequencing"
    """
    header_mapping = {
        "Required knowledge and skills": "QA_Knowledge-and-skills",
        "Key vocabulary terms": "QA_Key-Vocabulary",
        "Common misconceptions or challenges": "QA_Common-misconceptions",
        "Vocabulary Terms": "VA_Vocabulary-Terms",
        "Definitions": "VA_Definitions",
        "Special Attention": "VA_Special-Attention",
        "Teaching Approaches": "IR_Teaching-Approaches",
        "Scaffolding": "IR_Scaffolding",
        "Sequencing": "IR_Sequencing"
    }
    
    parsed_result = {value: "" for value in header_mapping.values()}
    header_pattern = re.compile(r'^\s*(?P<header>.+?):\s*$', re.IGNORECASE)
    current_key = None
    
    for line in output_text.splitlines():
        line = line.strip()
        if not line:
            continue
        
        header_match = header_pattern.match(line)
        if header_match:
            header_text = header_match.group("header").strip()
            for expected_header, csv_key in header_mapping.items():
                if header_text.lower() == expected_header.lower():
                    current_key = csv_key
                    break
            else:
                current_key = None
            continue
        
        if line.startswith("-") or line.startswith("*"):
            line = line[1:].strip()
        
        if current_key:
            if parsed_result[current_key]:
                parsed_result[current_key] += "\n" + line
            else:
                parsed_result[current_key] = line
    
    parsed_result = {k: v.strip() for k, v in parsed_result.items()}
    return parsed_result

def test_api_connection():
    """
    Test the API connection and return diagnostic information.
    """
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return False, "API key not found in environment variables. Please check your .env file."
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Try to list available models as a simple connectivity test
        try:
            models = genai.list_models()
            model_names = [model.name for model in models]
            
            # Display full model names for clarity
            full_model_names = ", ".join(model_names[:5])
            
            # Extract short names for better readability
            short_model_names = []
            for name in model_names:
                if "/" in name:
                    short_name = name.split("/")[-1]
                    short_model_names.append(short_name)
                else:
                    short_model_names.append(name)
            
            short_names_str = ", ".join(short_model_names[:5])
            
            # Return with both full and shortened names for clarity
            return True, f"Connection successful. Found {len(model_names)} models.\n\nFull model names: {full_model_names}...\n\nShort names: {short_names_str}..."
        except Exception as e:
            return False, f"Connected to API but couldn't list models: {str(e)}"
            
    except Exception as e:
        return False, f"API connection test failed: {str(e)}"

def generate_analysis(full_prompt: str, model_name: str):
    # Check if the API key is properly loaded
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY environment variable not found. Please check your .env file.")
        return {}, ""
    
    # Display API key status without showing the actual key
    st.info(f"API key found: {'Yes' if api_key else 'No'} (length: {len(api_key) if api_key else 0})")
    
    try:
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Get available models
        available_models = genai.list_models()
        available_model_names = [model.name for model in available_models]
        
        # Extract the short model name from selection
        short_model_name = model_name.split("/")[-1] if "/" in model_name else model_name
        
        # Find a suitable model by looking for the short name in any of the available models
        found_model = None
        for avail_model in available_model_names:
            if short_model_name in avail_model or model_name in avail_model:
                found_model = avail_model
                st.info(f"Using model: {found_model}")
                break
        
        # If no matching model found, try to find a default text model
        if not found_model:
            st.warning(f"Model '{model_name}' not found. Looking for alternative text models...")
            
            # Prioritize models in this order: gemini-1.5-flash, gemini-1.5-pro, text-bison, chat-bison
            preferred_models = ["gemini-1.5-flash", "gemini-1.5-pro", "text-bison", "chat-bison"]
            
            for preferred in preferred_models:
                for avail_model in available_model_names:
                    if preferred in avail_model:
                        found_model = avail_model
                        st.info(f"Using alternative model: {found_model}")
                        break
                if found_model:
                    break
        
        # If still no model found, use the first available model
        if not found_model and available_model_names:
            found_model = available_model_names[0]
            st.warning(f"No preferred models found. Using first available model: {found_model}")
        
        if not found_model:
            st.error("No suitable models found in your account.")
            return {}, ""
        
        # Get the model
        model = genai.GenerativeModel(found_model)
        
        # Generate content
        with st.spinner("Generating analysis... This may take a minute."):
            output_text = ""
            try:
                # Use different generation config based on model type
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_output_tokens": 8192,  # More conservative token limit
                }
                
                response = model.generate_content(
                    full_prompt,
                    generation_config=genai.GenerationConfig(**generation_config)
                )
                
                output_text = response.text
                        
            except Exception as e:
                st.error(f"Error during text generation: {e}")
                return {}, ""
        
        parsed_result = parse_analysis_output(output_text)
        return parsed_result, output_text
        
    except Exception as e:
        st.error(f"Error initializing Gemini API: {str(e)}")
        return {}, ""

def main():
    st.title("Assessment Analysis with Google Gemini")
    st.write("Enter your assessment data below. Your input will be processed with hidden internal instructions to guide the analysis.")
    
    # API connection status check
    with st.expander("API Connection Status"):
        if st.button("Check API Connection"):
            success, message = test_api_connection()
            if success:
                st.success(message)
                
                # Get available models for the dropdown
                try:
                    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
                    models = genai.list_models()
                    
                    # Show available model choices
                    st.subheader("Available Models:")
                    for model in models:
                        model_name = model.name
                        display_name = model_name.split("/")[-1] if "/" in model_name else model_name
                        st.write(f"- {display_name} ({model_name})")
                        
                except Exception as e:
                    st.error(f"Couldn't list models: {e}")
            else:
                st.error(message)
                st.write("Troubleshooting tips:")
                st.write("1. Make sure you have a .env file with GEMINI_API_KEY=your_key")
                st.write("2. Check that dotenv is installed: pip install python-dotenv")
                st.write("3. Verify your API key is active in Google AI Studio")
                st.write("4. Check if you have the latest version of the Google Generative AI library:")
                st.code("pip install --upgrade google-generativeai")
    
    # Hidden internal instructions (not displayed to the user)
    internal_instructions = """
COSTAR Prompt for Assessment Analysis

Context:
You are analyzing a set of educational assessment questions for a lesson being developed. These questions will be used both during instruction (formative assessment) and as a summative assessment at the end of the lesson. The analysis will identify the key skills, knowledge areas, concepts, and essential vocabulary that students need to master in order to successfully answer these questions. Vocabulary development is a critical component of the lesson structure, with specific guidelines for implementation.

Objective:
>>>INPUTS
Unit Title: [Your Unit Title]
Lesson Title: [Your Lesson Title]
Learning Objective: [Your Learning Objective]
Associated Standard Code: [Standard Code]
Associated Standard Text: [Standard Text]
DOK Low MC Item 1: [Item 1]
DOK Low MC Item 2: [Item 2]
DOK Low MC Item 3: [Item 3]
DOK Medium MC Item 1: [Item 1]
DOK Medium MC Item 2: [Item 2]
DOK Medium MC Item 3: [Item 3]
DOK Medium MC Item 4: [Item 4]
DOK Medium MC Item 5: [Item 5]
DOK High MC Item 1: [Item 1]
DOK High MC Item 2: [Item 2]
DOK High MC Item 3: [Item 3]
Constructed Response Prompt: [Prompt]
Constructed Response Sample Answer: [Sample Answer]

>>>TASK
Using the information above, analyze the provided assessment questions to:
1. Identify all knowledge areas, skills, and concepts required to answer each question.
2. Determine the prerequisite knowledge students would need.
3. Identify key vocabulary terms (4-6 words) that are essential for understanding the lesson content.
4. Create a comprehensive outline of content that must be covered in the lesson (Warmup, Instruction, and Summary sections).
5. Highlight any potential knowledge gaps or challenging areas that will require special attention.

Style:
- Analytical and thorough.
- Use bullet points to list specific skills, knowledge points, and vocabulary terms.

Tone:
- Educational and practical.
- Focus on pedagogical implications.

Audience:
- Educators and instructional designers.

Response:
Provide your analysis in the following format:

Question Analysis Summary:
- Required knowledge and skills:
- Key vocabulary terms:
- Common misconceptions or challenges:

Vocabulary Analysis:
- Vocabulary Terms:
- Definitions:
- Special Attention:

Implementation Recommendations:
- Teaching Approaches:
- Scaffolding:
- Sequencing:

Note: Focus only on the analysis of the assessment questions.
    """
    
    # Text area for the user to enter the data to be analyzed.
    user_input = st.text_area("Enter your assessment data", value="", height=300)
    
    # Get model names from environment or use defaults
    try:
        # Try to get actual available models
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        models = genai.list_models()
        model_options = []
        
        # Look for preferred models first
        preferred_types = ["gemini-1.5-flash", "gemini-1.5-pro", "text-bison", "chat-bison"]
        for preferred in preferred_types:
            for model in models:
                if preferred in model.name:
                    model_options.append(model.name)
                    break
        
        # If no preferred models found, add the first few available models
        if not model_options and models:
            model_options = [model.name for model in models[:5]]
            
    except Exception:
        # Fallback model options if we can't get actual models
        model_options = [
            "gemini-1.5-flash", 
            "gemini-1.5-pro", 
            "models/text-bison-001", 
            "models/chat-bison-001"
        ]
    
    # Add a default option if no models found
    if not model_options:
        model_options = ["gemini-1.5-flash"]
    
    # Model selection dropdown with discovered models
    selected_model = st.selectbox("Select Model", model_options)
    
    if st.button("Generate Assessment Analysis"):
        if not user_input.strip():
            st.error("Please enter your assessment data before generating analysis.")
            return
        
        # Combine the hidden internal instructions (with helper text) with the user's input.
        helper_text = "[INTERNAL INSTRUCTIONS APPENDED]\n"
        full_prompt = helper_text + internal_instructions + "\n" + user_input
        
        # Check for API key before proceeding
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            st.error("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
            
            # Show instructions for setting up API key
            with st.expander("How to set up your API key"):
                st.write("1. Go to https://ai.google.dev/ to get a Gemini API key")
                st.write("2. Create a file named `.env` in your project directory")
                st.write("3. Add this line to the file: `GEMINI_API_KEY=your_actual_api_key_here`")
                st.write("4. Restart this Streamlit application")
            return
        
        parsed_result, full_output = generate_analysis(full_prompt, selected_model)
        
        if full_output:
            st.subheader("Full Model Output")
            st.text_area("Output", full_output, height=300)
            
            st.subheader("Parsed Analysis")
            st.table(parsed_result)
            
            csv_headers = [
                "QA_Knowledge-and-skills",
                "QA_Key-Vocabulary",
                "QA_Common-misconceptions",
                "VA_Vocabulary-Terms",
                "VA_Definitions",
                "VA_Special-Attention",
                "IR_Teaching-Approaches",
                "IR_Scaffolding",
                "IR_Sequencing"
            ]
            
            csv_buffer = io.StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerow(parsed_result)
            csv_data = csv_buffer.getvalue().encode('utf-8')
            
            st.download_button(
                label="Download Analysis as CSV",
                data=csv_data,
                file_name="assessment_analysis.csv",
                mime="text/csv",
            )
        else:
            st.error("No output was generated. Please check your API key and input details.")

if __name__ == "__main__":
    main()