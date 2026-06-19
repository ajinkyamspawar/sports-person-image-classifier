import streamlit as st
import util
import tempfile
import os

st.set_page_config(
    page_title="Sports Person Classifier",
    page_icon="🏆",
    layout="centered"
)


util.load_saved_artifacts()

st.title("🏆 Sports Person Image Classifier🏆")

st.write(
    "Upload an image of a sports personality. "
    "The model uses face detection, wavelet transforms, "
    "and an SVM classifier to identify the athlete. "
    "**You can also try your own image for fun and see who the model thinks you resemble! 😄**" 

    "*Don't worry we don't save any image*"
)

st.write(
    "*Use .jpg or .png files only*"
    
)


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(uploaded_file)

    if st.button("Predict"):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name

        try:
            result = util.classify_image(temp_path)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if result is not None:

            st.subheader("🏅 Predicted Player")
            st.success(result["player"])

            st.subheader("🎯 Confidence")
            st.write(f"{result['confidence']} %")
            st.progress(result['confidence'] / 100)
        else:
            st.error("No face with two eyes detected in the image.")

st.markdown("---")
st.markdown(
    "📂 GitHub Repository: https://github.com/ajinkyamspawar/sports-person-image-classifier"
)