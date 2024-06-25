import gradio as gr

class App:
    def create_interface(self):
        with gr.Blocks() as demo:

            
            with gr.Tab("Heatmaps"):
                gr.Markdown("""Upload the lesion image and visualize""")
                with gr.Column():
                    

               

            
            with gr.Tab("Example-Based XAI"):
                
        self.demo_interface = demo

    def launch(self):
        self.demo_interface.launch()

app = App()
app.launch()