# 🧠 Vietnamese Medical Chatbot using RAG (Retrieval-Augmented Generation)

![Hugging Face Space](https://img.shields.io/badge/HF-Model-blue?logo=huggingface)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![LoRA Fine-tuning](https://img.shields.io/badge/LoRA-Fine--tuned-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-yellow)
![Gradio](https://img.shields.io/badge/Gradio-Deployed-red?logo=gradio)

## 📌 Overview

This project is a **Vietnamese Medical Chatbot** built using the **Retrieval-Augmented Generation (RAG)** . It combines:

- 🔍 Dense retrieval using **FAISS** with a fine-tuned **SBERT** (LoRA) model
- 🧠 Answer generation using **ViT5** (fine-tuned with QLoRA)
- 📦 Deployable on **Hugging Face Spaces** using **Gradio**

> 🏥 Designed to help users ask Vietnamese medical questions and receive contextual answers backed by relevant documents.

---

## 📌 System Architecture

<p align="center">
  <img src="https://github.com/hdbhoang2703/vietnamese_medical_chatbot/blob/main/assets/rag_pipeline.png" width="700"/>
</p>

**1. Input Question → 2. Embedding & Retrieval → 3. Generation → 4. Answer**

- **Retriever**: FAISS index with medical corpus, vectorized using fine-tuned SBERT
- **Generator**: ViT5 model fine-tuned with QLoRA to generate fluent, context-aware answers
- **UI**: Gradio deployed to Hugging Face Spaces

---

## 🛠️ Technical Stack & Skills

| Category        | Tool/Frameworks                                |
|----------------|-------------------------------------------------|
| Language Model | [ViT5-large](https://huggingface.co/VietAI/vit5-large), QLoRA |
| Embedding      | [SBERT](https://huggingface.co/keepitreal/vietnamese-sbert),LoRA     |
| Vector Search  | [FAISS](https://github.com/facebookresearch/faiss) |
| Chunking       | SentenceSplitter, Custom Preprocessing          |
| UI             | [Gradio](https://www.gradio.app/)            |
| Deployment     | Hugging Face Spaces                             |


---

## 🚀 Demo

🔗 **Try it live on Hugging Face Spaces:**  
[https://huggingface.co/spaces/baohoang2734/vietnamese-medical-chatbot](https://huggingface.co/spaces/baohoang2734/vietnamese-medical-chatbot)

<p align="center">
  <img src="https://raw.githubusercontent.com/hdbhoang2703/vietnamese_medical_chatbot/main/assets/demo_chatbot.png" width="1000"/>
</p>


---


