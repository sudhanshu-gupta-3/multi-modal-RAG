import gradio as gr

PDF_PATH = '/content/drive/MyDrive/multimodal_rag/docs/sample.pdf'

def dummy_search(question, image):
    answer = 'Demo answer. Connect your RAG pipeline here.'
    source = 'Source PDF: ' + PDF_PATH
    return answer, source

with gr.Blocks() as demo:
    gr.Markdown('# Multi-modal RAG Demo')
    q = gr.Textbox(label='Question')
    img = gr.Image(type='pil', label='Image (optional)')
    btn = gr.Button('Search')
    out1 = gr.Textbox(label='Answer')
    out2 = gr.Textbox(label='Source')
    btn.click(dummy_search, inputs=[q, img], outputs=[out1, out2])

demo.launch(server_name='0.0.0.0', server_port=7860)
