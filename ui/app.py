import os
import streamlit as st
import requests
from PIL import Image
import io

# Page Configuration
st.set_page_config(
    page_title="Casting Defect Detection App",
    page_icon="⚙️",
    layout="centered"
)

# FastAPI Backend Endpoint URL
API_URL = "http://127.0.0.1:8000/predict"
# FastAPI Backend Endpoint URL (use service name inside docker network, fallback to localhost)
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.title("⚙️ Casting Defect Detection System")
st.markdown(
    "Upload a casting image to analyze whether it is **ok** or **defective** "
    "using the deployed FastAPI deep learning model."
)

# Model selection dropdown to match backend Enum (ModelChoice)
model_choice = st.selectbox(
    "Select Model Architecture",
    options=["transfer_learning", "custom_cnn"],
    format_func=lambda x: "Transfer Learning (MobileNetV2)" if x == "transfer_learning" else "Custom CNN"
)

# File uploader widget for images
uploaded_file = st.file_uploader(
    "Choose a casting image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Casting Image", width=100)
    
    if st.button("🔍 Predict Defect", type="primary"):
        with st.spinner("Analyzing image via FastAPI backend..."):
            try:
                # Convert image to bytes to send via multipart form-data
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format if image.format else "JPEG")
                img_byte_arr = img_byte_arr.getvalue()
                
                files = {"file": (uploaded_file.name, img_byte_arr, uploaded_file.type)}
                data = {"model_name": model_choice}
                
                # Send POST request to FastAPI backend with both files and form data
                response = requests.post(API_URL, files=files, data=data)
                
                if response.status_code == 200:
                    res_data = response.json()
                    score = res_data.get("prediction")
                    result = res_data.get("result")
                    confidence = res_data.get("confidence")
                    model_used = res_data.get("model_used")
                    
                    st.success("Analysis Complete!")
                    
                    # Display Results in Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if result == "ok":
                            st.metric(label="Status", value="✅ OK")
                        else:
                            st.metric(label="Status", value="❌ Defective")
                    with col2:
                        st.metric(label="Prediction Score", value=f"{score:.4f}")
                    with col3:
                        st.metric(label="Confidence", value=f"{confidence:.2f}%")
                        
                    st.caption(f"Processed using model: `{model_used}`")
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the FastAPI server. Please make sure `serving/api.py` is running.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


                # Add this section to your ui/app.py to display model details, prediction history, and performance metrics from the logs

