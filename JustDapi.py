import streamlit as st
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology
import numpy as np
from collections import defaultdict
import pandas as pd

# --- Session state ---
if "clear_uploads" not in st.session_state:
    st.session_state.clear_uploads = False

# --- Page config and style ---
st.set_page_config(page_title="DAPI Cell Counter", page_icon="🔬", layout="wide")
st.markdown(
    """
    <style>
        body {
            background-image: url('https://pbs.twimg.com/media/GmuT894XoAAGm5a?format=jpg&name=4096x4096');
            background-size: cover;
            color: white;
        }
        .stApp {
            background: rgba(255, 182, 193, 0.8);
            padding: 20px;
            border-radius: 15px;
        }
        h1, h2, h3 {
            color: #ff69b4;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("DAPI Cell Counting Web App 🔬")

# --- Clear uploads button ---
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🗑️ Clear All Files"):
        st.session_state.clear_uploads = True
        st.experimental_rerun()

# --- Upload TIFF files ---
uploaded_files = st.file_uploader(
    "Upload DAPI TIFF files",
    type=["tif"],
    accept_multiple_files=True,
    key=None if not st.session_state.clear_uploads else str(np.random.rand())
)
st.session_state.clear_uploads = False  # reset

# --- Sample key extractor ---
def extract_sample_key(filename):
    return filename.replace(".tif", "").split("/")[-1]

# --- Process uploaded DAPI images ---
results = []

for file in uploaded_files:
    sample = extract_sample_key(file.name)
    dapi_image = tiff.imread(file)
    
    # DAPI cell segmentation
    dapi_mask = morphology.remove_small_objects(dapi_image > filters.threshold_otsu(dapi_image), min_size=10)
    dapi_labels = measure.label(dapi_mask)
    dapi_count = len(measure.regionprops(dapi_labels))

    results.append({
        "Sample": sample,
        "DAPI+ Cells": dapi_count
    })

with st.expander(f"🔬 Results for {sample}"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Raw DAPI in grayscale
    ax[0].imshow(dapi_image, cmap='gray')
    ax[0].set_title("Raw DAPI")
    ax[0].axis('off')
    
    # Segmented DAPI in blue
    blue_mask = np.zeros((*dapi_mask.shape, 3))
    blue_mask[..., 2] = dapi_mask.astype(float)
    ax[1].imshow(blue_mask)
    ax[1].set_title("Segmented DAPI+ Cells")
    ax[1].axis('off')

    plt.tight_layout()
    st.pyplot(fig)

# --- Summary table ---
if results:
    st.subheader("📊 Summary Table of DAPI Cell Counts")
    df_results = pd.DataFrame(results)
    st.dataframe(df_results)

    # CSV download
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", data=csv, file_name="dapi_cell_counts.csv", mime="text/csv")
else:
    if uploaded_files:
        st.warning("⚠️ No valid DAPI images processed.")
    else:
        st.info("Please upload DAPI TIFF files to get started.")
