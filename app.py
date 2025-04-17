import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow import argmax
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
from streamlit_image_select import image_select


st.set_page_config(
    page_title="Retinal Disease Detection",
    page_icon = ":eye:",
    initial_sidebar_state = 'auto'
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with st.sidebar:
        st.image('bg1.jpg')
        st.title("Retinal Detection")
        st.subheader("Detection of diseases present in the Retinal.This helps an user to easily detect the disease and provide Grad-CAM Visualization")

st.write("""
         # Retinal Disease Detection
         """
         )

model = load_model('Modeleye.h5')

labels= ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

def grad_cam(fname):
    DIM = 224
    img = tf.keras.preprocessing.image.load_img(fname, target_size=(DIM, DIM))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer('conv5_block16_concat')
        iterate = tf.keras.models.Model([model.input], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(x)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((7, 7))

    img = tf.keras.preprocessing.image.load_img(fname)
    img = tf.keras.preprocessing.image.img_to_array(img)
    alpha=0.4
    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img * 0.5
    img1 = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    col2.image(img1, caption='Grad cam')#,use_column_width="always")
    #st.image(img1)


def predict(image_file):
  img = tf.keras.preprocessing.image.load_img(image_file, target_size=(224, 224))
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  #img_array = img_array / 255.0
  img_batch = np.expand_dims(img_array, axis=0)
  predictions = model.predict(img_batch)
  predicted_class = argmax(predictions[0])
  return labels[predicted_class]

upload_img = st.file_uploader("", type=["jpg", "png"])

file = image_select(
    label="Please upload or select an image file",
    images=[
        "Images/10015_left.jpg",
        "Images/1020_left.jpg",
        "Images/1034_left.jpg",
        "Images/112_right.jpg",
    ],
)
b1, b2, b3, b4, b5, b6, b7= st.columns(7)
b1.button("Reset", type="primary")
class_btn = b7.button("Classify")

if class_btn:
    st.markdown("# Result")
    col1, col2 = st.columns(2)
    if file is None and upload_img is None:
        st.text("Please upload or select an image file")
    else:
        if upload_img is not None:
            col1.image(upload_img.read(), caption='Original Image')
            file=upload_img
        else:
            col1.image(file, caption='Original Image')
        prediction = predict(file)
        grad_cam(file)
        
        string = "Detected Disease : " + prediction
        if prediction == 'Normal':
            st.balloons()
            st.sidebar.success(string)
            st.markdown("## Normal")

        elif prediction == 'Cataract':
            st.sidebar.warning(string)
            st.markdown("## Cataract")
            st.info("Cataract is a clouding of the eye's lens, resulting in blurred vision. Grad-CAM highlights areas affected by cataracts, typically focusing on the lens region in retinal images.")

        elif prediction == 'Diabetic Retinopathy':
            st.sidebar.warning(string)
            st.markdown("## Diabetic Retinopathy")
            st.info("Diabetic retinopathy is a complication of diabetes that damages the blood vessels in the retina, leading to vision impairment. Grad-CAM highlights areas with abnormal blood vessel growth or leakage in retinal images")

        elif prediction == 'Glaucoma':
            st.sidebar.warning(string)
            st.markdown("## Glaucoma")
            st.info("Glaucoma is a group of eye disorders that damage the optic nerve, often due to increased pressure in the eye. Grad-CAM highlights the optic nerve region in retinal images, where damage may indicate the presence of glaucoma.")


