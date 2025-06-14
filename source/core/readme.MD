# 🧠 Vietnamese Medical Chatbot using RAG (Retrieval-Augmented Generation)

![Hugging Face Space](https://img.shields.io/badge/HF-Space-blue?logo=huggingface)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![LoRA Fine-tuning](https://img.shields.io/badge/LoRA-Fine--tuned-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)

## 📌 Overview

This project is a **Vietnamese Medical Chatbot** built using the **Retrieval-Augmented Generation (RAG)** framework. It combines:

- 🔍 Dense retrieval using **FAISS** with a fine-tuned **SBERT** (LoRA) model
- 🧠 Answer generation using **ViT5** (fine-tuned with QLoRA)
- 📦 Deployable on **Hugging Face Spaces** using **Streamlit**

> 🏥 Designed to help users ask Vietnamese medical questions and receive contextual answers backed by relevant documents.

---

## 📌 System Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/yourusername/yourrepo/main/assets/rag_pipeline.png" width="700"/>
</p>

**1. Input Question → 2. Embedding & Retrieval → 3. Generation → 4. Answer**

- **Retriever**: FAISS index with medical corpus, vectorized using fine-tuned SBERT
- **Generator**: ViT5 model fine-tuned with QLoRA to generate fluent, context-aware answers
- **UI**: Streamlit App deployed to Hugging Face Spaces

---

## 🛠️ Technical Stack & Skills

| Category        | Tool/Frameworks                                |
|----------------|-------------------------------------------------|
| Language Model | [ViT5](https://huggingface.co/VietAI/vit5-large), QLoRA |
| Embedding      | SBERT (with LoRA), [sentence-transformers]      |
| Vector Search  | [FAISS](https://github.com/facebookresearch/faiss) |
| Chunking       | SentenceSplitter, Custom Preprocessing          |
| UI             | [Streamlit](https://streamlit.io/)              |
| Deployment     | Hugging Face Spaces                             |
| Format         | Markdown, Python, YAML                          |

---

## 🚀 Demo

🔗 **Try it live on Hugging Face Spaces:**  
👉 [https://huggingface.co/spaces/yourusername/vietnamese-medical-chatbot](https://huggingface.co/spaces/yourusername/vietnamese-medical-chatbot)

<p align="center">
  <img src="https://raw.githubusercontent.com/yourusername/yourrepo/main/assets/demo_screenshot.png" width="700"/>
</p>

---

## 📁 Directory Structure

