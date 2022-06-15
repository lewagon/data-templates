import streamlit as st
from PIL import Image
import requests

# Set page tab display
st.set_page_config(
   page_title="Simple Image Uploader",
   page_icon= ':framed-picture:',
   layout="wide",
   initial_sidebar_state="expanded",
)

# Set your API url
url = 'http://localhost:8000'


# App title and description
st.header('Simple Image Uploader 📸')
st.markdown('''
            > This is a Le Wagon boilerplate for any data science projects that involve exchanging images between a Python API and a simple web frontend.

            > **What's here:**

            > * [Streamlit](https://docs.streamlit.io/) on the frontend
            > * [FastAPI](https://fastapi.tiangolo.com/) on the backend
            > * [PIL/pillow](https://pillow.readthedocs.io/en/stable/) and [opencv-python](https://github.com/opencv/opencv-python) for working with images
            > * Backend and frontend can be deployed with Docker
            ''')

st.markdown("---")

### Create a native Streamlit file upload input
img_file_buffer = st.file_uploader('Upload an image')

if img_file_buffer is not None:

  col1, col2 = st.columns(2)

  with col1:
    ### Display the image user uploaded
    st.image(Image.open(img_file_buffer), caption="Here's the image you uploaded ☝️")

  with col2:
    with st.spinner("Wait for it..."):
      ### Get bytes from the file buffer
      img_bytes = img_file_buffer.getvalue()

      ### Make request to API (stream=True to stream response as bytes)
      res = requests.post(url + "/upload_image", files={'img': img_bytes}, stream=True)

      if res.status_code == 200:
        ### Display the image returned by the API
        st.image(res.content, caption="Image returned from API ☝️")
      else:
        print(res.status_code, res.content)

