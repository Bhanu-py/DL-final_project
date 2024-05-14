import streamlit as st
import tensorflow
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import os
import numpy as np
import keras
# from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import inception_v3
from model import orientation_model, damage_model

# tensorflow gpu
physical_devices = tensorflow.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))


st.title("Vehicle Orientation and quality Assessment")

st.markdown("### 🚗🚘🚔🚖🚍 Classification Application")
st.markdown("This application predicts the orientation of the Vehicle out of 9 Orientations relative to Driver side and also the detects physical damage on the vehicle.")
menu = ["Select image from the below list", "Upload From Computer"]
choice = st.sidebar.radio(label="Menu", options=["Select image from the below list", "choose your own image"])
#
if choice == "Select image from the below list":
    file = st.sidebar.selectbox("choose your image", os.listdir("examples"))
    uploaded_file = os.path.join(os.getcwd(), "examples", file)
else:
    uploaded_file = st.sidebar.file_uploader("Please upload an image:", type=['jpeg', 'jpg', 'png'])

# Loading model
### By full model
# model = keras.models.load_model('model/model_optimized')
### By Weights

@st.cache_resource
def load_model(type='orientation'):
    if type == 'orientation':
        model = orientation_model(n_classes=9)
        model.load_weights('transfer-learning\\orientation_detection\model\\top_model_weights2.h5')
    elif type == 'damage':
        model = damage_model(n_classes=2, fc_layer_size=512)
        model.load_weights('transfer-learning\\damage_detection\\model\\damage_detection_weights2.h5')
    return model

orient_m= load_model('orientation')
damage_m = load_model('damage')


orientation_classes = {'Driver side front': 0,
                        'Driver side rear': 1,
                        'Passenger side front': 2,
                        'Passenger side rear': 3,
                        'Driver side': 4,
                        'Front': 5,
                        'Passenger side': 6,
                        'Rear': 7,
                        'Unknown': 8}

damage_classes = {'Damage':0, 'No Damage':1}

def predict_orientation(img_file):
    img = Image.open(img_file)
    new_img = np.asarray(img)

    image = load_img(img_file, color_mode="rgb", target_size=(224, 224), interpolation="nearest")
    image_array = keras.preprocessing.image.image_utils.img_to_array(image)
    image_array = keras.applications.inception_v3.preprocess_input(image_array, data_format=None)
    input_arr = np.array([image_array])  # Convert single image to a batch.
    
    # Label and score prediction based on pre-trained random forest model
    pred = orient_m.predict(input_arr)
    class_label = {i for i in orientation_classes if orientation_classes[i] == pred.argmax(axis=1)}
    proba = np.max(pred)
    print(pred)
    print(class_label, proba)

    # Label and Score formatting
    pred_class_label = ''.join(class_label)
    pred_score = round(proba, 4)
    return pred_class_label, pred_score
# pred_class_label, pred_score = predict('Images/12397.jpg')

def predict_damage(img_file):
    img = Image.open(img_file)
    new_img = np.asarray(img)

    image = load_img(img_file, color_mode="rgb", target_size=(224, 224), interpolation="nearest")
    image_array = keras.preprocessing.image.image_utils.img_to_array(image)
    image_array = keras.applications.inception_v3.preprocess_input(image_array, data_format=None)
    input_arr = np.array([image_array])  # Convert single image to a batch.
    
    # Label and score prediction based on pre-trained random forest model
    pred = damage_m.predict(input_arr)
    class_label = {i for i in damage_classes if damage_classes[i] == pred.argmax(axis=1)}
    proba = np.max(pred)
    print(pred)
    print(class_label, proba)

    # Label and Score formatting
    pred_class_label = ''.join(class_label)
    pred_score = round(proba, 4)
    return pred_class_label, pred_score



if uploaded_file is not None:
    orient_pred_class_label, orient_pred_score = predict_orientation(uploaded_file)
    damage_pred_class_label, damage_pred_score = predict_damage(uploaded_file)
    # Display image
    st.image(Image.open(uploaded_file), caption="Uploaded image", use_column_width=True)
    st.write("**Orientation:**", orient_pred_class_label)
    st.write("**Probability:**", f'{round(orient_pred_score*100)}%')
    # space
    st.write("")
    st.write("**Damage:**", damage_pred_class_label)
    st.write("**Probability:**", f'{round(damage_pred_score*100)}%')


    expander = st.expander("For more details !!")
    expander.write({"Orientation": orient_pred_class_label,
                    "Probability": orient_pred_score})
    
    expander.write({"Damage": damage_pred_class_label,
                    "Probability": damage_pred_score})
