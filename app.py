import os
import sys
import gradio as gr
from pathlib import Path

# Add the project directory to the Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Set Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_app.settings')

# Import Django after setting up the environment
import django
django.setup()

# Import your RAG functionality
from rag_app.rag_utils import process_query
from rag_app.views import get_documents

# Create a Gradio interface
def rag_query(query):
    results = process_query(query)
    return results

# Build a simple interface
with gr.Blocks() as demo:
    gr.Markdown("# RAG Document Retrieval System")
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(label="Enter your query")
            submit_btn = gr.Button("Search")
        
        with gr.Column():
            output = gr.Textbox(label="Results")
    
    submit_btn.click(fn=rag_query, inputs=query_input, outputs=output)

if __name__ == "__main__":
    demo.launch()
