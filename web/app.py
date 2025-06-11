import streamlit as st
from source.rag_pipeline import RAGpipline
from source.embedding_model import EmbeddingModel
import os

def main():
    rag = RAGpipline()
    
    faiss_path = "data/faiss/faiss_index.faiss"

    if not os.path.exists(faiss_path):
        rag.embedding_data("data/Corpus_RAG", 200)
    else:
        print("bug1") 

    st.title('About Presight')

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input('Message')

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        answer = rag.run(prompt, top_k=10)
        st.chat_message('assistant').markdown(answer)
        st.session_state.messages.append({'role': 'assistant', 'content': answer})
        
if __name__ == "__main__":
    main()