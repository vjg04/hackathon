
import streamlit as st
from PIL import Image
import os
import numpy as np
# import face_recognition
# import cv2
import pandas as pd
import pickle
from numpy import dot
from numpy.linalg import norm
# import dlib
import numpy as np

import streamlit as st
from streamlit_image_select import image_select
import os


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

Image_Name = None
result_img = None

def object_detection11(img_path):
    
    """
    Object Detection (On Image) From TF2 Saved Model
    =====================================
    """

    # from google.colab.patches import cv2_imshow

    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # PROVIDE PATH TO IMAGE DIRECTORY
    # IMAGE_PATHS = './training/images/upload/glass124.jpg'
    IMAGE_PATHS = img_path
    # IMAGE_PATHS = './content/training_demo/images/upload/paper593.jpg'


    # PROVIDE PATH TO MODEL DIRECTORY
    # PATH_TO_MODEL_DIR = 'D:\Hack\Hackathon\\training\exported_models\my_model'
    PATH_TO_MODEL_DIR = '../../training/exported_models/my_model'
    # PATH_TO_MODEL_DIR = '/content/training_demo/exported_models/my_model'

    # PROVIDE PATH TO LABEL MAP
    # PATH_TO_LABELS = 'D:\Hack\Hackathon\\training\\annotations\label_map.pbtxt'
    PATH_TO_LABELS = '../../training/annotations/label_map.pbtxt'
    # PATH_TO_LABELS = '/content/training_demo/annotations/label_map.pbtxt'

    # PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
    MIN_CONF_THRESH = float(0.6)

    # LOAD THE MODEL



    PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

    print('Loading model...', end='')
    start_time = time.time()

    # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    # LOAD LABEL MAP DATA FOR PLOTTING

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)



    def load_image_into_numpy_array(path):
        """Load an image from file into a numpy array.
        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.
        Args:
        path: the file path to the image
        Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
        """
        return np.array(Image.open(path))




    print('Running inference for {}... '.format(IMAGE_PATHS), end='')

    image = cv2.imread(IMAGE_PATHS)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_with_detections = image.copy()

    # SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=MIN_CONF_THRESH,
        agnostic_mode=False)

    print('Done')
    # # DISPLAYS OUTPUT IMAGE
    # cv2_imshow(image_with_detections)
    # # CLOSES WINDOW ONCE KEY IS PRESSED
    # # white paper 


    return image_with_detections




# from detection import object_detection11

# import cv2
# from mtcnn import MTCNN


# streamlit page

st.set_page_config(
    page_title='HACKTHON - Garbage Detection Dashbosrd',
    layout="wide",
    initial_sidebar_state="expanded",
)


html_temp = """
<div style="background-color:#2F396F;padding:0.7px">
<h3 style="color:white;text-align:center;" >HACKTHON - Garbage Detection Dashbosrd</h3>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)

# st.image('https://i.imgur.com/3XqSI3B.png', width=180)

col1, col2= st.columns([5,1])

with col1:
    # st.image('hackthon.png', width=180, height=120)

    image = Image.open("hackthon.png")
    image = image.resize((180, 120))  # Set the desired width and height
    st.image(image)

with col2:
    image = Image.open("nagarro.png")
    image = image.resize((180, 120))  # Set the desired width and height
    st.image(image)


path = "../../training/images/upload"

# Get a list of all the image filenames in the folder, with full file paths
image_paths = [os.path.join(path, filename) for filename in os.listdir(path) if filename.endswith(".jpg") or filename.endswith(".png")]

# Set the default index of the image filenames list
default_index = 0

# Create an expandable section with a button to show/hide the image selector
with st.expander("Select From Sample Image"):
    # if st.button("▲ Show Image Selector"):
        # Display the image selector
        # selected_image_path = image_select(image_paths, default_index=default_index)
    selected_image_path = image_select(label= "", images = image_paths, index=default_index)

    Image_Name = selected_image_path

    # result_image = object_detection11(selected_image_path)

    # st.image(result_image)


    # # Load the selected image
    # selected_image = Image.open(selected_image_path)
    # if "selected_image_path" in st.session_state:
    #     # Load the selected image
    #     selected_image = Image.open(st.session_state.selected_image_path)

    #     # Display the selected image
    #     # st.image(selected_image)

    #     # Remove the selected image path from session state
    #     del st.session_state["selected_image_path"]



st.markdown("""---""")


col1, col2= st.columns(2)


with col1:

    uploaded_image = st.file_uploader('Choose An Image')
    if uploaded_image != None:
        
        display_image = Image.open(uploaded_image)
        path = './uploads/testimg.jpg'
        display_image.save(path)
        Image_Name = path

        # st.header('Your Uploaded Image')

        # st.image(display_image,width = 300)
        # st.image(uploaded_image)


with col2:

    st.markdown("""
    <style>
    .stButton button {
        margin: 1.4rem 1rem;
        padding: 1.2rem 1rem;
        background-color: #5cb85c;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    if st.button('Run Algorithm'):
        if Image_Name != None:
            with st.spinner('Algorithm Running.....'):
                result_img = object_detection11(Image_Name)
            # st.image(result_img)
        pass

col1, col2= st.columns(2)

with col1:
    if Image_Name!=None and result_img is not None:
        st.image(Image_Name)

with col2:
    if result_img is not None:
        st.image(result_img)

