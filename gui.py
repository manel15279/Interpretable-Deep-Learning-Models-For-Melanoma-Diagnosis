import gradio as gr
import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend as K
import tensorflow.keras
from tensorflow.python.framework import ops
import gc
from tensorflow.keras.models import load_model
from utils import read_and_preprocess_img, preprocess_image, superimpose
from gradcam import generate_grad_cam_heatmap
from gradcampp import grad_cam_plus
from scorecam import ScoreCam
from tensorflow.keras.applications.mobilenet import preprocess_input


def confidence_change_metric(model, img_array, heatmap, original_confidence):
    heatmap_expanded = np.expand_dims(heatmap, axis=-1)
    heatmap_expanded = np.repeat(heatmap_expanded, 3, axis=-1)
    perturbed_img_array = img_array * heatmap_expanded
    perturbed_confidence = model.predict(perturbed_img_array)[0][0]
    if perturbed_confidence < 0.5:
        perturbed_confidence = 1 - perturbed_confidence
    confidence_change = perturbed_confidence - original_confidence
    return confidence_change



def get_heatmaps(img_path, layer_name, htm, model_name):
    print(img_path)
    confs = []
    model=modelld
    img_array = read_and_preprocess_img(model_name, img_path, (256, 256))

    pred = model.predict(img_array)
    if pred[0][0] > 0.5:
        c = 'mel'
    else:
        c = 'not_mel'
    #print("predicted class : ", c, pred)

    confidence = pred[0][0] if c == 'mel' else 1 - pred[0][0]

    original = image.load_img(img_path, target_size=(256, 256))

    #GRADCAM
    if 'GradCAM' in htm:
        original_img = np.uint8((img_array[0] - img_array[0].min()) / (img_array[0].max() - img_array[0].min()) * 255)
        grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.output])
        hmconf, melanoma_heatmap = generate_grad_cam_heatmap(model, grad_model, img_array)
        hif = 0.5
        superimposed_img_gc = melanoma_heatmap * hif + original_img * (1 - hif)
        superimposed_img_gc = np.minimum(superimposed_img_gc, 255.0).astype(np.uint8)
        hmconf = np.expand_dims(hmconf, axis=0)

        confidence_change1 = confidence_change_metric(model, img_array, hmconf[0], confidence)
        confs.append(confidence_change1)

    if 'GradCAM++' in htm:
        #GRADCAM++  # Replace with the actual layer name you want to visualize
        img = preprocess_image(model_name, img_path)
        heatmap_plus = grad_cam_plus(model, img, img_array, layer_name)
        grad_cam_pp_superimposed = superimpose(img_path, heatmap_plus)
        heatmap_plus = cv2.resize(heatmap_plus, (256, 256)) # or cv2.COLOR_RGB2GRAY depending on the input format
        heatmap_plus = np.expand_dims(heatmap_plus, axis=0)

        confidence_change2 = confidence_change_metric(model, img_array, heatmap_plus[0], confidence)
        confs.append(confidence_change2)

    #SCORECAM
    if 'ScoreCAM' in htm:
        score_cam = ScoreCam(model, img_array, layer_name)
        score_cam_superimposed = superimpose(img_path, score_cam)
        score_cam = cv2.resize(score_cam, (256, 256)) # or cv2.COLOR_RGB2GRAY depending on the input format
        score_cam = np.expand_dims(score_cam, axis=0)
    
        confidence_change3 = confidence_change_metric(model, img_array, score_cam[0], confidence)
        confs.append(confidence_change3)

    plt.figure(figsize=(16, 5))

    # Plot original image
    plt.subplot(1, len(htm), 1)
    #plt.title(f'Original Image\nTrue Class: {true_class}\nPredicted Class: {c} (Conf: {confidence:.4f})')
    plt.imshow(original)
    plt.axis('off')

    if 'GradCAM' in htm:
        # Plot Grad-CAM
        index = htm.index('GradCAM')
        plt.subplot(1, len(htm), index+1)
        #plt.title(f'Grad-CAM\nConfidence Change: {confidence_change1:.4f}')
        plt.imshow(superimposed_img_gc)
        plt.axis('off')

    if 'GradCAM++' in htm:
        # Plot Grad-CAM++
        index = htm.index('GradCAM++')
        plt.subplot(1, len(htm), index+1)
        #plt.title(f'Grad-CAM++\nConfidence Change: {confidence_change2:.4f}')
        plt.imshow(grad_cam_pp_superimposed)
        plt.axis('off')

    if 'ScoreCAM' in htm:
        # Plot Score-CAM
        index = htm.index('ScoreCAM')
        plt.subplot(1, len(htm), index+1)
        #plt.title(f'Score-CAM\nConfidence Change: {confidence_change3:.4f}')
        plt.imshow(score_cam_superimposed)
        plt.axis('off')
    
    cax = plt.axes([0.92, 0.25, 0.01, 0.49])
    colorbar = plt.colorbar(cax=cax)
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(['not_mel', 'mel'])
    plt.subplots_adjust(right=0.9)
    plt.savefig("heatmaps\\heatmap.png")
    plot = ["heatmaps\\heatmap.png"]
    plt.close()
    
    return c, confidence, confs, plot



with gr.Blocks() as demo:
    with gr.Tab("Heatmaps"):
        gr.Markdown("Upload the lesion image and visualize")
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        model = gr.Dropdown(["ResNet50", "DenseNet201", "MobileNet", "Xception", "InceptionV3", "VGG19", "VGG16"], multiselect=False, label="Models", info="Select a model : ")
                        layer = gr.Dropdown(choices=[],interactive=True, multiselect=False, label="Layers", info="Select a layer of the model : ")
                    htm = gr.CheckboxGroup(['GradCAM', 'GradCAM++', 'ScoreCAM'], label='Visualization Method', info='Select one or more methods to visualize the heatmaps : ')
                    img = gr.Image(label="Upload an Image", type="filepath")
                    btn = gr.Button("Explain")
                    def get_model_layers(model_name):
                        global modelld 
                        model_dict = {
                            "ResNet50": "models\modelresnet50 (1).keras",
                            "DenseNet201": 'models\modeldensenet201.keras',
                            "MobileNet": 'models\modelmobilenet.keras',
                            "Xception": 'models\modelxception (2).keras',
                            "InceptionV3": 'models\modelinceptionv3.keras',
                            "VGG19": 'models\modelvgg19.keras',
                            "VGG16": 'models\modelvgg16.keras'
                        }
                        model_path = model_dict[model_name]
                        modelld = load_model(model_path) 
                        layers = [layer.name for layer in modelld.layers]
                        return gr.update(choices=layers, value=None)

                    model.change(fn=get_model_layers, inputs=model, outputs=[layer])

                with gr.Column():
                    gallery = [gr.Gallery(label="Heatmap(s)", columns=(1,2))]
                    with gr.Row():
                        pred_cls = gr.Textbox(label="Predicted Class")
                        conf_pred = gr.Textbox(label="Confidence")
                    with gr.Row():
                        conf1 = gr.Textbox(label="Confidence Change GradCAM")
                        conf2 = gr.Textbox(label="Confidence Change GradCAM++")
                        conf3 = gr.Textbox(label="Confidence Change ScoreCAM")
                btn.click(fn=get_heatmaps, inputs=[img]+[layer]+[htm]+[model], outputs=[pred_cls]+[conf_pred]+[conf1, conf2, conf3]+gallery)

    with gr.Tab("Example-Based XAI"):
        gr.Markdown("Example-based explanations with siamese networks")


if __name__ == "__main__":
    demo.launch()
