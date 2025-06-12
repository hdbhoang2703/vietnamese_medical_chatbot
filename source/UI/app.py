import streamlit as st
from source.core.rag_pipeline import RAGpipeline
from source.core.embedding_model import EmbeddingModel
import os
import time

def main():
    rag = RAGpipeline()
    
    faiss_path = "data/faiss/faiss_index.faiss"

    if not os.path.exists(faiss_path):
        rag.embedding_data("data/Corpus_RAG", 200)  

    st.title('Vietnamese Medical Chatbot')

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input('Message')

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        with st.chat_message('assistant'):
            message_placeholder = st.empty()
            
            # Loading animation
            full_response = ""
            
            for i in range(5):
                loading_dots = "." * ((i % 3) + 1)
                message_placeholder.markdown(f"Đang xử lý{loading_dots}")
                time.sleep(0.3)
            
            # Generate answer
            answer = rag.run(prompt, 5)
            
            words = answer.split()
            for i, word in enumerate(words):
                full_response += word + " "
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.05) 
            
            message_placeholder.markdown(full_response)
            
        st.session_state.messages.append({'role': 'assistant', 'content': answer})
        
if __name__ == "__main__":
    main()