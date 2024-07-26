import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array



def read_and_preprocess_img(model, path, size):
    img = load_img(path, target_size=size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    if model == 'ResNet50':
        from tensorflow.keras.applications.renset50 import preprocess_input
    if model == 'DenseNet201':
        from tensorflow.keras.applications.densenet import preprocess_input
    if model == 'MobileNet':
        from tensorflow.keras.applications.mobilenet import preprocess_input
    if model == 'Xception':
        from tensorflow.keras.applications.xception import preprocess_input
    if model == 'InceptionV3':
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    if model == 'VGG19':
        from tensorflow.keras.applications.vgg19 import preprocess_input
    if model == 'VGG16':
        from tensorflow.keras.applications.vgg16 import preprocess_input
    x = preprocess_input(x)
    return x

def superimpose(original_img_path, cam, emphasize=False):
    
    img_bgr = cv2.imread(original_img_path)
    img_bgr = cv2.resize(img_bgr, (256, 256))

    heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    hif = .5
    superimposed_img = heatmap * hif + img_bgr * (1-hif)
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255  
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return superimposed_img_rgb

def preprocess_image(model, img_path, target_size=(256, 256)):
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    if model == 'ResNet50':
        from tensorflow.keras.applications.renset50 import preprocess_input
    if model == 'DenseNet201':
        from tensorflow.keras.applications.densenet import preprocess_input
    if model == 'MobileNet':
        from tensorflow.keras.applications.mobilenet import preprocess_input
    if model == 'Xception':
        from tensorflow.keras.applications.xception import preprocess_input
    if model == 'InceptionV3':
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    if model == 'VGG19':
        from tensorflow.keras.applications.vgg19 import preprocess_input
    if model == 'VGG16':
        from tensorflow.keras.applications.vgg16 import preprocess_input
    img = preprocess_input(img)
    return img


