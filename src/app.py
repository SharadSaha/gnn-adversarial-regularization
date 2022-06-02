import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import os


st.markdown(
    '''# Adversarial attacks and graph regularization

Demonstration of adversarial attack on images from the MNIST dataset,
and use of graph regularization to build models that are robust to such attacks.


## Adversarial attack using Fast Gradient Sign Method (FGSM)

FGSM can be used to create adversarial patterns or pertubations which can 
combined with the original image to attain a visually similar image which
is capable of tricking the model to misclassify the image.'''
)


original_mnist_image = Image.open(os.path.join('src','images','original_image.png'))
perturbation_image = Image.open(os.path.join('src','images','perturbation.png'))
final_image = Image.open(os.path.join('src','images','final_image.png'))

col1, col2, col3 = st.columns(3)

with col1:
    st.image(original_mnist_image,caption="Original image")

with col2:
    st.image(perturbation_image,caption="Adversarial pattern")

with col3:
    st.image(final_image,caption="Final image")


st.markdown("-----")
st.markdown("## Misclassification on addition of adversarial pattern")

img1 = Image.open(os.path.join('src','images','adv Epsilon = 0.010.png'))
img2 = Image.open(os.path.join('src','images','adv Epsilon = 0.040.png'))
img3 = Image.open(os.path.join('src','images','adv Epsilon = 0.050.png'))
img4 = Image.open(os.path.join('src','images','adv Epsilon = 0.090.png'))
img5 = Image.open(os.path.join('src','images','adv Epsilon = 0.150.png'))


col1, col2, col3 = st.columns(3)

with col1:
    st.image(img1)
    st.image(img4)

with col2:
    st.image(img2)
    st.image(img5)

with col3:
    st.image(img3)


st.markdown('''<hr style="border:2px solid gray">''',unsafe_allow_html=True)
st.markdown('''
    
## Increasing robustness of CNN model using graph regularization
---
### Dataset
---

''')

image1 = Image.open(os.path.join('src','images','img1.png'))
image2 = Image.open(os.path.join('src','images','img2.png'))
image3 = Image.open(os.path.join('src','images','img3.png'))
image4 = Image.open(os.path.join('src','images','img4.png'))
image5 = Image.open(os.path.join('src','images','img5.png'))
image6 = Image.open(os.path.join('src','images','img6.png'))


col1, col2, col3 = st.columns(3)

with col1:
    st.image(image1)
    st.image(image4)

with col2:
    st.image(image2)
    st.image(image5)

with col3:
    st.image(image3)
    st.image(image6)

st.markdown("---")
st.markdown("### Comparison - Normal model vs graph regularized model")
st.markdown("---")
st.image(os.path.join('src','images','history.png'))
