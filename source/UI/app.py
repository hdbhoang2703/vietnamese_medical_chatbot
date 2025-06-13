import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import gradio as gr
from source.core.rag_pipeline import RAGpipeline
from source.core.embedding_model import EmbeddingModel
import time

def main():
    rag = RAGpipeline()
    
    faiss_path = "data/faiss/faiss_index.faiss"

    if not os.path.exists(faiss_path):
        rag.embedding_data("data/Corpus_RAG", 200)

    def rag_response(query):
        start_time = time.time()
        answer = rag.run(query, 3)  
        end_time = time.time()
        return f"{answer}"

    interface = gr.Interface(
        fn=rag_response,
        inputs=gr.Textbox(label="Nhập câu hỏi y tế:"),
        outputs=gr.Textbox(label="Kết quả:"),
        title="Demo - Vietnamese Medical Chatbot"
    )
    interface.launch()

if __name__ == "__main__":
    main()