
import streamlit as st
from textclassification import classify_text
from database import save_submission
from PIL import Image
from imagecaption import generate_caption

st.header("Hello User, Upload or Enter Text and We Will Classify It For You")

classifier=st.selectbox("Choose Classifier:",["Text", "Image"] )

if classifier=="Text":
    
    with st.form("text_form"):
        text=st.text_input("Enter Text")
        submitted = st.form_submit_button("Classify")

    if submitted:
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            labels, scores = classify_text(text)
            save_submission(text, labels)  

            st.write("Predicted labels:", labels or "No toxic labels predicted")

if classifier == "Image":
    st.subheader("Classify an image")

    with st.form("image_form"):
        uploaded_image = st.file_uploader("Upload an image",type=["jpg", "jpeg", "png"])
        image_submitted = st.form_submit_button("Classify image")

    if image_submitted:
        if uploaded_image is None:
            st.warning("Please upload an image.")
        else:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, width=400)

            with st.spinner("Generating caption and classifying..."):
                caption = generate_caption(image)

                labels, scores = classify_text(caption)

                save_submission(caption, labels)

                st.write("Generated caption:", caption)
                st.write("Predicted labels:", labels or "No toxic labels predicted")
                st.write("Probabilities:", scores)



