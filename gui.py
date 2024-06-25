import gradio as gr

class App:
    def __init__(self) -> None:
        self.create_interface()
        
    def create_interface(self):
        with gr.Blocks() as demo:

            
            with gr.Tab("Heatmaps"):
                gr.Markdown("""Upload the lesion image and visualize""")
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            model = [gr.Dropdown(["ResNet50", "DenseNet201", 'MobileNet', 'Xception', 'InceptionV3', 'VGG19', 'VGG16'], multiselect=False, label="Models", info="Select a model : ")]
                            layer = [gr.Dropdown([(f"{layer.name}", i) for i, layer in enumerate(model.layers)], multiselect=False, label="Layers", info="Select a layer of the model : ")]
                            htm = [gr.Checkbox(['GradCAM', 'GradCAM++', 'ScoreCAM'], label='Visualization Method', info='Select one or more methods to visualize the heatmaps :')]
                        with gr.Column():
                            image = gr.Image(label="Upload an Image", type="filepath")
               

            
            with gr.Tab("Example-Based XAI"):
                
        self.demo_interface = demo

    def launch(self):
        self.demo_interface.launch()

app = App()
app.launch()