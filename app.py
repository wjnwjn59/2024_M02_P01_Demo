import os
import base64
import streamlit as st

from models.naive_bayes.train import model
from components.streamlit_footer import footer

@st.cache_data(max_entries=1000)
def inference_and_display_result(text):
    clf_prediction = model.predict(text)
    st.markdown('**Classification result**')
    st.write(f'Input: {text}')
    st.write(f'Prediction: {clf_prediction}')

def main():
    st.set_page_config(
        page_title="AIO2024 Module02 Project Text Classification - AI VIETNAM",
        page_icon='static/aivn_favicon.png',
        layout="wide"
    )

    col1, col2 = st.columns([0.8, 0.2], gap='large')
    
    with col1:
        st.title('AIO2024 - Module02 - Text Project')
        st.title(':sparkles: :blue[Naive Bayes] Spam Text Classification Demo')
        
    with col2:
        logo_img = open("static/aivn_logo.png", "rb").read()
        logo_base64 = base64.b64encode(logo_img).decode()
        st.markdown(
            f"""
            <a href="https://aivietnam.edu.vn/">
                <img src="data:image/png;base64,{logo_base64}" width="full">
            </a>
            """,
            unsafe_allow_html=True,
        )

    uploaded_msg = st.text_input('Input your message (Press enter to run)', placeholder='Hello AIO..')

    st.divider()

    example_button = st.button('Use an example')

    if example_button:
        inference_and_display_result('I love talking about you')
    elif uploaded_msg:
        inference_and_display_result(uploaded_msg)

    footer()


if __name__ == '__main__':
    main()