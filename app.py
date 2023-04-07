import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import streamlit as st
#from io import StringIO

st.title("Potato Disease Prediction Application🥔")
st.subheader("Developed by Group Q8🪄")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    with st.columns(3)[1]:
        st.image(uploaded_file,width=300, caption='Enter any caption here')


test_image = tf.keras.utils.load_img(uploaded_file,target_size=(256,256))


st.subheader(":red[Prediction:] ")
model=load_model('potatoes.h5')

class_names=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
classes=['Early Blight Disease', 'Late Blight Disease', 'healthy leaf']

test_image = tf.keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)

# Result array
result = model.predict(test_image)
confidence=round(100 * (np.max(result)), 2)
prediction=(classes[np.argmax(result)])
st.subheader(prediction)
"_Confidence_= ",confidence,'%'
