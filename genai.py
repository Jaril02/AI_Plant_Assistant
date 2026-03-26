import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from weather.weather import get_weather, watering_advice
from PIL import Image
from app.pdf_generator import PDFReport
import time
from databases.database import init_db, add_user, add_schedule, get_schedules
from datetime import datetime
from weather.notify import send_sms



# Load environment variables and configure Gemini AI
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)
api_key=os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Load class names and disease information
raw_class_indices = json.load(open('scripts/class_indices.json'))
class_indices = {str(v): k for k, v in raw_class_indices.items()} 
disease_info = json.load(open('scripts/plant_disease_info.json'))

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

def set_modern_ui():
    st.markdown(
        """
        <style>
        /* Modern UI Styling */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Custom CSS for modern look */
        .main-header {
            background: linear-gradient(90deg, #2E8B57, #3CB371);
            padding: 2rem;
            border-radius: 20px;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            text-align: center;
            color: white;
        }
        
        .main-header h1 {
            font-size: 3rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .main-header p {
            font-size: 1.2rem;
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        

        .ai-card {
            background: #adadff; /* very light bluish background */
            border: 1px solid #e0e0f0;
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            transition: transform 0.2s ease;
        }

        .ai-card:hover {
            transform: scale(1.02);
        }

        
        .ai-light-card {
            background: rgba(255, 255, 255, 0.95);
            color: #333;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            line-height: 1.6;
        }

        .upload-area {
            border: 3px dashed #4CAF50;
            border-radius: 20px;
            padding: 3rem;
            text-align: center;
            background: rgba(76, 175, 80, 0.1);
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            border-color: #45a049;
            background: rgba(76, 175, 80, 0.2);
            transform: translateY(-2px);
        }
        
        .prediction-card {
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        }
        
        .disease-info {
            background: linear-gradient(135deg, #2196F3, #1976D2);
            color: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        }
        
        .ai-section {
            background: linear-gradient(135deg, #9C27B0, #7B1FA2);
            color: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        }
        
        .sidebar {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.2rem;
            margin: 0.5rem;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        }
        
        .stButton > button {
            background: linear-gradient(90deg, #4CAF50, #45a049);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        
        .stTextInput > div > div > input {
            border-radius: 15px;
            border: 2px solid #e0e0e0;
            padding: 0.75rem 1rem;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #4CAF50;
            box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1);
        }
        
        .stFileUploader > div > div {
            border-radius: 15px;
            border: 2px solid #e0e0e0;
            transition: all 0.3s ease;
        }
        
        .stFileUploader > div > div:hover {
            border-color: #4CAF50;
        }
        
        .tab-content {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .feature-icon {
            font-size: 2rem;
            margin-right: 1rem;
        }
        
        .loading-animation {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #4CAF50;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .success-animation {
            animation: successPulse 0.6s ease-in-out;
        }
        
        @keyframes successPulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .footer {
            text-align: center;
            padding: 2rem;
            color: white;
            background: rgba(0,0,0,0.1);
            border-radius: 20px;
            margin-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_modern_ui()

# Function to interact with Gemini AI
def ask_gemini(question):
    try:
        model = genai.GenerativeModel(
            model_name="gemini-flash-latest",
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

st.set_page_config(
    layout="wide", 
    page_title="🌱 AI Plant Assistant", 
    page_icon="🌱",
    initial_sidebar_state="expanded"
)

phone_number = st.text_input(
    "Enter your phone number (with country code, e.g., +91XXXXXXXXXX):",
    key="user_phone"
)




# Main App
def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🌱 AI Plant Assistant</h1>
            <p>Advanced AI-powered plant disease detection and analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div class="sidebar">
                <h2>🚀 Features</h2>
                <p><span class="feature-icon">🔍</span>AI Disease Detection</p>
                <p><span class="feature-icon">🌦️</span>Weather & Watering Advice</p>
                <p><span class="feature-icon">📄</span>PDF Reports</p>
                <p><span class="feature-icon">💬</span>AI Chat Support</p>
            </div>
        """, unsafe_allow_html=True)
        city = st.text_input("Enter your city", value="Delhi", key="weather_city")

        if st.button("Check Weather and Notify",key="check_weather_btn"):
            weather = get_weather(city)
            
            if "error" in weather:
                st.error(weather["error"])
            else:
                st.info(f"🌡️ Temp: {weather['temp']}°C | 💧 Humidity: {weather['humidity']}% | {weather['description'].capitalize()}")
                advice = watering_advice(weather)
                st.success(advice)
                
                # Trigger SMS if plant needs watering
                if "Water" in advice or "No rainfall" in advice:
                    if phone_number:
                        sms_sent = send_sms(
                            phone_number,
                            f"🌱 Plant Watering Alert!\nWeather: {weather['description']}, Temp: {weather['temp']}°C, Humidity: {weather['humidity']}%\nAdvice: {advice}"
                        )
                        if sms_sent:
                            st.success("✅ Watering SMS sent successfully!")
                    else:
                        st.warning("⚠️ Please enter your phone number to receive SMS alerts.")


        
        st.markdown("### 📋 Settings")
        
        # Multi-select dropdown for features
        selected_features = st.multiselect(
            "Enable Features",
            ["PDF Report", "AI Chat", "Image Enhancement", "Historical Data"],
            default=["PDF Report", "AI Chat"],
            help="Select which features to enable"
        )
        
        # Dropdown with custom formatting
        report_format = st.selectbox(
            "Report Format",
            ["PDF", "HTML", "Text"],
            format_func=lambda x: f"📄 {x}",
            help="Choose your preferred report format"
        )
        
        # Dropdown with index
        confidence_threshold = st.selectbox(
            "Confidence Threshold",
            [0.5, 0.7, 0.8, 0.9, 0.95],
            index=2,  # Default to 0.8
            help="Minimum confidence level for disease detection"
        )
        
        st.markdown("---")
        
        # How to Use Dropdown
        with st.expander("📱 How to Use", expanded=False):
            st.markdown("""
                **Step-by-Step Instructions:**
                
                **1. 📸 Upload Plant Image**
                • Take a clear photo of the affected plant area
                • Ensure good lighting and focus
                • Upload JPG, JPEG, or PNG files only
                
                **2. 🔍 Get Instant Diagnosis**
                • AI will analyze your image automatically
                • Results show disease type and confidence level
                • Get treatment recommendations instantly
                
                **3. 📄 Download Detailed Report**
                • Generate comprehensive PDF reports
                • Includes disease details, treatment plans
                • Save for future reference
                
                **4. 💬 Ask AI for More Info**
                • Chat with AI about plant care
                • Get personalized advice
                • Ask follow-up questions
                
                **💡 Tips for Best Results:**
                • Use high-quality images
                • Include both healthy and affected areas
                • Ensure proper lighting conditions
                • Upload images of leaves, stems, or fruits
            """)
    
    # Main Content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main Upload Section
        st.markdown("""
            <div class="card">
                <h2>📸 Upload Plant Image</h2>
                <p>Get instant AI-powered disease diagnosis for your plants</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Plant Type Selection Dropdown
        plant_category = st.selectbox(
            "🌿 Plant Category",
            ["All Plants", "Vegetables", "Fruits", "Grains", "Ornamentals", "Herbs"],
            help="Select the category of your plant for better analysis"
        )
        
        # Disease Focus Dropdown
        if plant_category != "All Plants":
            disease_focus = st.multiselect(
                "🎯 Focus Areas",
                ["Leaf Diseases", "Root Diseases", "Stem Diseases", "Fruit Diseases", "Fungal Infections", "Bacterial Infections"],
                default=["Leaf Diseases"],
                help="Select specific disease types to focus on"
            )
        
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of the plant leaf or affected area"
        )
        
        if uploaded_image is not None:
            # Image Display
            st.markdown("""
                <div class="card">
                    <h3>🖼️ Uploaded Image</h3>
                </div>
            """, unsafe_allow_html=True)
            
            image = Image.open(uploaded_image)
            col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
            
            with col_img2:
                st.image(image, caption="Your Plant Image", use_container_width=True)
            
            # Classification Button
            if st.button('🔍 Analyze Plant Disease', use_container_width=True,key="analyze_btn"):
                if model is not None:
                    with st.spinner('🤖 AI is analyzing your plant...'):
                        prediction = predict_image_class(model, uploaded_image, class_indices)
                        st.session_state.prediction = prediction
                        st.session_state.disease_data = get_disease_info(prediction)
                        st.session_state.image = image
                        time.sleep(1)  # Simulate processing
                        
                        # Show selected options
                        st.info(f"🌿 Plant Category: {plant_category}")
                        if plant_category != "All Plants" and 'disease_focus' in locals():
                            st.info(f"🎯 Focus Areas: {', '.join(disease_focus)}")
                        st.info(f"📊 Confidence Threshold: {confidence_threshold}")


            st.title("🌱 Plant Care Scheduler")
            with st.form("user_form"):
                name = st.text_input("Name")
                phone = st.text_input("Phone (+91...)")
                if st.form_submit_button("Add User"):
                    init_db()
                    add_user(name, phone)
                    st.success("✅ User added!")

            with st.form("task_form"):
                user_id = st.number_input("User ID", min_value=1)
                task = st.text_input("Task (e.g., Watering)")
                task_time = st.text_input("Time (HH:MM)", value="07:00")
                frequency = st.selectbox("Frequency", ["daily", "weekly", "monthly"])
                day = st.text_input("Day (if weekly/monthly)")
                if st.form_submit_button("Add Task"):
                    init_db()
                    add_schedule(user_id, task, task_time, frequency, day)

                    st.success("✅ Task scheduled successfully!")

                    st.success("Task scheduled!")

            st.write("📋 Current Schedules:", get_schedules())


            if "prediction" in st.session_state:
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.session_state.prediction = prediction
                disease_data = get_disease_info(prediction)
                st.session_state.disease_data = disease_data
                
                st.success(f'✅ Analysis Complete!')

                # Prediction Result
                st.markdown(f"""
                    <div class="prediction-card">
                        <h3>🎯 Disease Detected</h3>
                        <h2>{prediction}</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                # Disease Information
                if disease_data:
                    st.markdown("""
                        <div class="disease-info">
                            <h3>📋 Disease Information</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        st.info(f"**🌿 Plant:** {disease_data.get('plant_name', 'N/A')}")
                        st.info(f"**⚠️ Symptoms:** {disease_data.get('symptoms', 'N/A')}")
                        st.info(f"**🔍 Causes:** {disease_data.get('causes', 'N/A')}")
                    
                    with info_col2:
                        st.info(f"**🛡️ Prevention:** {disease_data.get('preventive_measures', 'N/A')}")
                        st.info(f"**💊 Treatment:** {disease_data.get('treatment', 'N/A')}")
                else:
                    st.warning("📚 Local data not found. Generating AI-powered analysis...")
                    
                    # AI Analysis
                    with st.spinner('🤖 AI is generating detailed information...'):
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
                        
                        st.markdown("""
                            <div class="ai-card">
                                <h3>🤖 AI-Generated Analysis</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                            <div class="ai-light-card">
                                {ai_summary}
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Additional AI insights
                        ai_detailed = ask_gemini(f"Tell me more about {prediction} in detail")
                        ai_prevention = ask_gemini(f"What are the detailed prevention techniques for {prediction}?")
                        
                        st.markdown("""
                            <div class="ai-card">
                                <h3>🔬 Detailed Analysis</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                            <div class="ai-light-card">
                                {ai_detailed}
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("""
                            <div class="ai-card">
                                <h3>🛡️ Prevention Guide</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                            <div class="ai-light-card">
                                {ai_prevention}
                            </div>
                        """, unsafe_allow_html=True)

                        # 📄 Generate PDF Button
                        if st.button("📄 Generate PDF Report", use_container_width=True,key="pdf_btn"):
                            with st.spinner("Generating PDF..."):
                                try:
                                    # Save uploaded image
                                    safe_img = image.convert("RGB")
                                    image_path = "temp_uploaded_image.jpg"
                                    safe_img.save(image_path, format="JPEG")

                                    # Prepare disease data
                                    disease_data_for_pdf = {
                                        "Plant Name": st.session_state.disease_data.get("plant_name", st.session_state.prediction),
                                        "Symptoms": st.session_state.disease_data.get("symptoms", "Not found"),
                                        "Causes": st.session_state.disease_data.get("causes", "Not found"),
                                        "Preventive Measures": st.session_state.disease_data.get("preventive_measures", "Not found"),
                                        "Treatment": st.session_state.disease_data.get("treatment", "Not found"),
                                    }

                                    st.session_state["ai_summary"] = ai_summary
                                    st.session_state["ai_detailed"] = ai_detailed
                                    st.session_state["ai_prevention"] = ai_prevention


                                    # Weather & watering advice
                                    if "weather" in st.session_state:
                                        weather_data = st.session_state["weather"]
                                        weather_text = watering_advice(weather_data)
                                    else:
                                        weather_text = "Weather data unavailable."


                                    # Generate PDF
                                    pdf = PDFReport(title="Plant Disease Report")
                                    pdf.add_page()
                    
                                    pdf.add_image_and_text(
                                        image_path,
                                        disease_data_for_pdf,
                                        weather_info=weather_text,
                                        ai_summary=ai_summary,
                                        ai_detailed=ai_detailed,
                                        ai_prevention=ai_prevention,
                                    )
                                    pdf_path = pdf.export_pdf("reports/plant_disease_report.pdf")

                                    # Store PDF bytes in session state ✅
                                    with open(pdf_path, "rb") as f:
                                        st.session_state["pdf_bytes"] = f.read()

                                        st.success("✅ PDF report generated successfully!")

                                except Exception as e:
                                    st.error(f"❌ PDF generation failed: {e}")

                        # 📥 Persistent download button ✅
                        if "pdf_bytes" in st.session_state:
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=st.session_state["pdf_bytes"],
                                file_name="plant_disease_report.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                key="pdf_download_btn"
                            )



    
    with col2:
        # Quick Actions
        st.markdown("""
            <div class="card">
                <h3>⚡ Quick Actions</h3>
                <p>Get instant help and information</p>
            </div>
        """, unsafe_allow_html=True)
        
        # AI Chat Section
        st.markdown("""
            <div class="ai-section">
                <h3>💬 Ask AI Assistant</h3>
                <p>Get instant answers about plant care</p>
            </div>
        """, unsafe_allow_html=True)
        
        user_question = st.text_input(
            "Ask about plants, diseases, or care tips:",
            key="chat_input",
            placeholder="e.g., How to prevent tomato blight?"
        )
        
        if st.button('🤖 Ask AI', use_container_width=True,key="ask_ai_btn"):
            if user_question:
                with st.spinner('🤖 AI is thinking...'):
                    ai_answer = ask_gemini(user_question)
                    st.markdown("""
                        <div class="card">
                            <h4>🤖 AI Response:</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    st.write(ai_answer)
            else:
                st.warning("⚠️ Please enter a question first.")
    
    # Footer
    st.markdown("""
        <div class="footer">    
            <p>🌱 Built with ❤️ using Streamlit & Google Gemini AI</p>
            <p>© 2025 Plant Disease AI Assistant | Empowering Farmers with AI</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()