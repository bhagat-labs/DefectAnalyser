import streamlit as st
import requests
from PIL import Image
import io
import os
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Casting Defect Detection App",
    page_icon="⚙️",
    layout="centered"
)

# FastAPI Backend Endpoint URL
API_URL = "http://127.0.0.1:8000/predict"

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
    try:
        # Display the uploaded image with reduced width
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
                    st.error(f"An error occurred during prediction: {str(e)}")
    except Exception as e:
        st.error(f"Could not open uploaded image: {str(e)}")

# ==========================================
# Model Comparison & History Dashboard
# ==========================================
st.markdown("---")
st.subheader("📊 Model Comparison & History Dashboard")

try:
    UI_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(UI_DIR)
    
    possible_paths = [
        os.path.join(ROOT_DIR, "logs", "predictions.csv"),
        os.path.join(UI_DIR, "logs", "predictions.csv"),
        os.path.join(ROOT_DIR, "serving", "logs", "predictions.csv"),
        os.path.abspath("logs/predictions.csv")
    ]
    
    LOG_FILE_PATH = None
    for path in possible_paths:
        if os.path.exists(path):
            LOG_FILE_PATH = path
            break

    if LOG_FILE_PATH and os.path.exists(LOG_FILE_PATH):
        df_logs = pd.read_csv(LOG_FILE_PATH)
        
        if df_logs is not None and not df_logs.empty:
            # 1. Detect column names dynamically
            cols = df_logs.columns.tolist()
            model_col = next((c for c in ["model_version", "model_used", "model"] if c in cols), None)
            score_col = next((c for c in ["score", "prediction", "confidence_score"] if c in cols), None)
            label_col = next((c for c in ["predicted_label", "predicted_class", "result"] if c in cols), None)
            
            # 2. Model Comparison Section
            if model_col and score_col:
                st.markdown("### 🔍 Model Comparison Matrix")
                agg_dict = {score_col: ['count', 'mean']}
                if label_col:
                    agg_dict[label_col] = 'count'
                
                comparison_df = df_logs.groupby(model_col).agg(agg_dict).reset_index()
                comparison_df.columns = [
                    "Model Architecture", 
                    "Total Inferences", 
                    f"Avg {score_col.replace('_', ' ').title()}"
                ] + ([f"Class Log Count"] if label_col else [])
                
                st.dataframe(comparison_df, use_container_width=True)
            else:
                st.info("Model comparison columns not found in logs yet. Make new predictions via FastAPI.")

            # 3. Paginated Recent Inferences (5 items per page)
            st.markdown("### 📈 Recent Inferences (Paginated)")
            
            items_per_page = 5
            total_rows = len(df_logs)
            total_pages = max(1, (total_rows + items_per_page - 1) // items_per_page)
            
            col_p1, col_p2 = st.columns([2, 2])
            with col_p1:
                page_number = st.selectbox(
                    "Select Page", 
                    options=list(range(1, total_pages + 1)), 
                    format_func=lambda x: f"Page {x} of {total_pages}"
                )
            
            start_idx = (page_number - 1) * items_per_page
            end_idx = start_idx + items_per_page
            
            paginated_df = df_logs.iloc[start_idx:end_idx]
            st.dataframe(paginated_df, use_container_width=True)
            st.caption(f"Showing rows {start_idx + 1} to {min(end_idx, total_rows)} of {total_rows} total log entries.")
            
        else:
            st.info("Log file is empty. Run predictions using the API to populate metrics.")
    else:
        st.info("No prediction logs found yet. Run predictions via the API to generate history logs.")
except Exception as e:
    st.warning(f"Error loading dashboard metrics: {str(e)}")