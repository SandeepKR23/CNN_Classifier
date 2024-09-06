import streamlit as st
from PIL import Images
import tensorflow as tf
import numpy as np

model = tf.keras.modela.load_model("model.h5")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    img = image.resize((244, 244))
    image_array = np.array(img)
    image_array = tf.expand_dims(image_array, axis=0)
    result = model.predict(image_array)
    argmax_index = np.argmax(result, axis=1)
    if argmax_index[0] == 0:
        st.image(image, caption="Predicted class: Cat")
    else:
        st.image(image, caption="Predicted class: Dog")