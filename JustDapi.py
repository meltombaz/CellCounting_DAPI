import streamlit as st
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology
import numpy as np
import pandas as pd

# --- Session state setup ---
if "upload_key" not in st.session_state:
    st.session_state.upload_key = str(np.random.rand())

# --- Page config and style ---
st.set_page_config(page_title="DAPI Cell Counter", page_icon="🔬", layout="wide")

# --- Title and Video ---
st.title("DAPI Cell Counting Web App 🔬")


st.markdown(
    """
    <div style='text-align: center;'>
        <iframe width="210" height="118" 
                src="https://youtu.be/ic8j13piAhQ?si=bdkkHZyMv_WMy8sh" 
                frameborder="0" allow="accelerometer; autoplay; clipboard-write; 
                encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen>
        </iframe>
    </div>
    """,
    unsafe_allow_html=True
)


# --- Clear uploads button ---
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🗑️ Clear All Files"):
        st.session_state.upload_key = str(np.random.rand())
        st.experimental_rerun()

# --- File uploader ---
uploaded_files = st.file_uploader(
    "Upload DAPI TIFF files",
    type=["tif"],
    accept_multiple_files=True,
    key=st.session_state.upload_key
)

# --- Display uploaded filenames ---
if uploaded_files:
    st.success(f"✅ Uploaded {len(uploaded_files)} DAPI file(s):")
    for f in uploaded_files:
        st.write(f"- {f.name}")

# --- Sample key extractor ---
def extract_sample_key(filename):
    return filename.replace(".tif", "").split("/")[-1]

# --- Process DAPI files ---
results = []

for file in uploaded_files:
    sample = extract_sample_key(file.name)
    dapi_image = tiff.imread(file)
    
    # Segment nuclei using Otsu thresholding and remove small objects
    dapi_mask = morphology.remove_small_objects(dapi_image > filters.threshold_otsu(dapi_image), min_size=10)
    dapi_labels = measure.label(dapi_mask)
    dapi_count = len(measure.regionprops(dapi_labels))

    results.append({
        "Sample": sample,
        "DAPI+ Cells": dapi_count
    })

    # --- Visualization ---
    with st.expander(f"🔬 Results for {sample}"):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        # Raw DAPI image
        ax[0].imshow(dapi_image, cmap='gray')
        ax[0].set_title("Raw DAPI")
        ax[0].axis('off')
        
        # Blue DAPI mask overlay
        blue_mask = np.zeros((*dapi_mask.shape, 3))
        blue_mask[..., 2] = dapi_mask.astype(float)
        ax[1].imshow(blue_mask)
        ax[1].set_title("Segmented DAPI+ Cells (Blue)")
        ax[1].axis('off')

        plt.tight_layout()
        st.pyplot(fig)

# --- Summary Table ---
if results:
    st.subheader("📊 Summary Table of DAPI Cell Counts")
    df_results = pd.DataFrame(results)
    st.dataframe(df_results)

    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", data=csv, file_name="dapi_cell_counts.csv", mime="text/csv")
else:
    if uploaded_files:
        st.warning("⚠️ No valid DAPI images processed.")
    else:
        st.info("Please upload DAPI TIFF files to get started.")
