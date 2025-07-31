import streamlit as st
import tensorflow
from PIL import Image
import numpy as np

st.write('Classified')
classes = ['cats', 'dogs']

@st.cache_resource
def load_model():
    model = tensorflow.keras.models.load_model("classified.h5")
    return model
model = load_model()

def preprocess(image:Image.Image):
    img = image.resize((128,128))
    img = np.array(img)/255
    img = img.reshape(1,128,128,3)
    return img
upload_f = st.file_uploader('Choose an image', type = ['jpg', 'jpeg','png'])
if upload_f:
    #st.write('File Uploaded')
    image=Image.open(upload_f)
    st.image(image,caption='Uploaded Image')
    processed=preprocess(image)
    pred=model.predict(processed)[0]
    class_index=np.argmax(pred)
    pred_class=classes[class_index]
    st.success(f'Prediction : {pred_class}')
