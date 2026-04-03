"""Myanmar LNP Dataset Explorer - Streamlit App."""

import streamlit as st
from api.data_loader import load_data
from api.preprocess import clean_text


st.set_page_config(
    page_title="Myanmar LNP Dataset",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Myanmar LNP Dataset Explorer")
st.markdown("Explore and analyze Myanmar language NLP datasets")


def load_sample_data():
    """Load sample data for demo."""
    return [
        {"text": "သတင်းသည်သင်တန်းစာသင်ပါး", "label": "news"},
        {"text": "လူမှုကွပ်ကဲပ်ပါပံုး", "label": "social"},
    ]


# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Explore", "Predict", "Stats"]
)


if page == "Explore":
    st.header("Explore Data")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload JSONL file",
        type=["jsonl", "json", "csv"]
    )
    
    if uploaded_file:
        # Save temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as f:
            f.write(uploaded_file.getvalue())
            temp_path = f.name
        
        df = load_data(temp_path)
        os.unlink(temp_path)
        
        st.write(f"Loaded **{len(df)}** records")
        
        # Show data
        st.dataframe(df.head(100))
        
        # Download
        st.download_button(
            "Download sample",
            data=df.head(10).to_json(orient="records", force_ascii=False),
            file_name="sample.jsonl",
            mime="application/jsonl"
        )
    else:
        st.info("Upload a file or use sample data below")
        
        if st.button("Load Sample Data"):
            st.session_state["df"] = load_sample_data()
            st.rerun()


if page == "Predict":
    st.header("Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_text = st.text_area(
            "Enter Myanmar text",
            height=100,
            placeholder="သတင်းသည်ကို ဖတ်ပါး..."
        )
        
        if st.button("Predict"):
            if input_text:
                cleaned = clean_text(input_text)
                st.success(f"Cleaned: {cleaned}")
            else:
                st.warning("Please enter text")
    
    with col2:
        st.info("Model loading coming soon!")
        
        # Label distribution
        st.subheader("Label Distribution")
        labels = {
            "news": 0.3,
            "social": 0.2,
            "literary": 0.15,
            "legal": 0.1,
            "technical": 0.25,
        }
        
        import pandas as pd
        
        st.bar_chart(pd.DataFrame(
            list(labels.items()),
            columns=["Label", "Confidence"]
        ).set_index("Label"))


if page == "Stats":
    st.header("Statistics")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", "1,234")
    with col2:
        st.metric("Labels", "10")
    with col3:
        st.metric("Avg Length", "156")
    with col4:
        st.metric("Categories", "8")
    
    # Chart
    st.subheader("Label Distribution")
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    labels = list(range(10))
    counts = np.random.randint(50, 200, 10)
    
    fig, ax = plt.subplots()
    ax.bar(labels, counts)
    ax.set_xlabel("Label ID")
    ax.set_ylabel("Count")
    st.pyplot(fig)