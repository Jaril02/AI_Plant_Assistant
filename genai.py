import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
from pdf_generator import PDFReport


# Load environment variables and configure Gemini AI
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load class names and disease information
raw_class_indices = json.load(open('class_indices.json'))
class_indices = {str(v): k for k, v in raw_class_indices.items()} 
disease_info = json.load(open('plant_disease_info.json'))

# Load the pre-trained model with error handling
try:
    model = tf.keras.models.load_model('trained_model/plant_disease_prediction_model.h5', compile=False)
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None  # Prevent further usage if loading fails

# Function to load and preprocess image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to predict the disease class
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    print(f"Predicted class index: {predicted_class_index}")
    print(f"Available class_indices keys: {list(class_indices.keys())}")
    return class_indices[str(predicted_class_index)]

# Function to get disease information
def get_disease_info(predicted_class):
    return disease_info.get(predicted_class, {})

def set_bg_color():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #99faaa; /* Background color */
            color: #333234; /* Default text color */
        }

        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #333234;
        }

        /* Label texts like 'Upload Image', 'Enter your question' */
        label, .css-1cpxqw2, .css-1n76uvr {
            color: #333234 !important;
            font-weight: 600;
        }

        /* Input text inside textbox */
        input, textarea {
            color: #333234 !important;
        }

        /* Optional: Change file uploader button text */
        .css-1cpxqw2, .css-1n76uvr {
            color: #333234 !important;
        }

        /* Optional: change other form element text */
        .stTextInput>div>div>input, 
        .stTextArea textarea, 
        .stFileUploader>div>div {
            color: #333234 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_color()


# Function to interact with Gemini AI
def ask_gemini(question):
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 1000,
                "response_mime_type": "text/plain",
            }
        )
        
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(question)
        
        return response.text
    except Exception as e:
        error_msg = str(e)

        # Check if it's a quota limit issue
        if "429" in error_msg or "quota" in error_msg.lower():
            return "AI-generated response is not available due to API usage limit. Please try again later."

        # General error fallback
        return "An error occurred while generating AI response."
st.set_page_config(layout="wide", page_title="Plant Disease Detector", page_icon="🌿")



# Streamlit App



with st.sidebar:
    st.title("🌱 Plant AI Assistant")
    st.info("Upload your plant image and get disease insights!")

tab1, tab2, tab3 = st.tabs(["Prediction", "AI Info", "Ask AI"])

with tab1:
    st.title("🌱 Plant AI Assistant")
    st.write(f"Prediction: {st.session_state.get('prediction', 'No prediction yet')}")

with tab2:
    st.write(st.session_state.get('ai_summary', 'AI info not available'))


with tab3:
    user_question = st.text_input("Ask a question")
    if st.button("Ask"):
        st.write(ask_gemini(user_question))



if model is not None:
    st.title('AI for Farmers: Plant Disease Classifier')

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns([1,2])

        with col1:
            resized_img = image.resize((350, 350))
            st.image(resized_img)

        with col2:
            if st.button('Classify'):
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f'Prediction: {prediction}')
                print(f"Available keys: {list(disease_info.keys())[:5]}")
                st.session_state.prediction = prediction
                disease_data = get_disease_info(prediction)
                st.session_state.disease_data = disease_data

                if disease_data:
                    st.write(f"**Plant Name:** {disease_data.get('plant_name')}")
                    st.write(f"**Symptoms:** {disease_data.get('symptoms')}")
                    st.write(f"**Causes:** {disease_data.get('causes')}")
                    st.write(f"**Preventive Measures:** {disease_data.get('preventive_measures')}")
                    st.write(f"**Treatment:** {disease_data.get('treatment')}")
                else:
                    st.warning("No data found in local JSON. Using Gemini AI to generate information...")
                    ai_summary = ask_gemini(f"""
                        Give detailed info about the plant disease: {prediction}.
                        Include:
                        - Plant Name
                        - Symptoms
                        - Causes
                        - Preventive Measures
                        - Treatment
                        Format it clearly.
                    """)
                    st.subheader("AI-Generated Disease Info")
                    st.write(ai_summary)

                     # Automatically query Gemini AI for more details about the disease
                    ai_initial_response = ask_gemini(f"Tell me more about {prediction} in detail")
                    st.subheader('AI Generated Detailed Information:')
                    st.write(ai_initial_response)

                    # Automatically ask AI about prevention techniques
                    ai_prevention_response = ask_gemini(f"What are the detailed prevention techniques for {prediction}?")
                    st.subheader('AI Generated Prevention Techniques:')
                    st.write(ai_prevention_response)
        if st.button('Download PDF Report'):
            if uploaded_image is not None and 'prediction' in st.session_state:
                prediction = st.session_state.prediction
                disease_data = st.session_state.disease_data
                image_path = "temp_uploaded_image.png"
                image.save(image_path)

                # Fetch AI info from session state or regenerate
                ai_summary = ask_gemini(f"Give detailed info about the plant disease: {prediction}.")
                ai_initial_response = ask_gemini(f"Tell me more about {prediction} in detail")
                ai_prevention_response = ask_gemini(f"What are the detailed prevention techniques for {prediction}?")

                # Prepare content
                disease_data_for_pdf = {
                    "Plant Name": disease_data.get("plant_name", prediction),
                    "Symptoms": disease_data.get("symptoms", "Not found"),
                    "Causes": disease_data.get("causes", "Not found"),
                    "Preventive Measures": disease_data.get("preventive_measures", "Not found"),
                    "Treatment": disease_data.get("treatment", "Not found"),
                }

                # Generate PDF
                pdf = PDFReport()
                pdf.add_page()
                pdf.add_image_and_text(
                    image_path,
                    disease_data_for_pdf,
                    ai_summary=ai_summary,
                    ai_detailed=ai_initial_response,
                    ai_prevention=ai_prevention_response
                )
                pdf_path = pdf.export_pdf()

                # Download button
                with open(pdf_path, "rb") as file:
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=file,
                        file_name="plant_disease_report.pdf",
                        mime="application/pdf"
                    )        
                

    
   

    # Separate section for further Q&A with Gemini AI
    st.subheader('Ask AI more questions about the disease')
    user_question = st.text_input('Enter your question:')
    if st.button('Ask AI'):
        if user_question:
            ai_answer = ask_gemini(user_question)
            st.write(f"**AI Answer (Gemini):** {ai_answer}")
        else:
            st.write("Please enter a question.")
st.markdown("""
    <hr style="border:1px solid #ccc"/>
    <p style='text-align: center;'>© 2025 Plant Disease AI App | Built with ❤️ using Streamlit & Gemini</p>
""", unsafe_allow_html=True)