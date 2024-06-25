import gradio as gr
from tensorflow.keras.models import load_model


with gr.Blocks() as demo:
    with gr.Tab("Heatmaps"):
        gr.Markdown("Upload the lesion image and visualize")
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    model = [gr.Dropdown(["ResNet50", "DenseNet201", "MobileNet", "Xception", "InceptionV3", "VGG19", "VGG16"], multiselect=False, label="Models", info="Select a model : ")]
                    layer = gr.Dropdown(choices=[],interactive=True, multiselect=False, label="Layers", info="Select a layer of the model : ")
                    htm = gr.CheckboxGroup(['GradCAM', 'GradCAM++', 'ScoreCAM'], label='Visualization Method', info='Select one or more methods to visualize the heatmaps : ')


                    def get_model_layers(model_name):
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
                        model = load_model(model_path) 
                        layers = [layer.name for layer in model.layers]
                        print(layers)
                        return gr.update(choices=layers, value=None)

                    model[0].change(fn=get_model_layers, inputs=model[0], outputs=[layer])

                with gr.Column():
                        image = gr.Image(label="Upload an Image", type="filepath")
                        pred_cls = gr.Textbox(label="Predicted Class")

    with gr.Tab("Example-Based XAI"):
        gr.Markdown("Example-based explanations with siamese networks")


if __name__ == "__main__":
    demo.launch()
