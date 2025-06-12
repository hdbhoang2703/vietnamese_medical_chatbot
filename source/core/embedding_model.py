from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, PeftConfig
from sentence_transformers import SentenceTransformer, models
from typing import List
import numpy as np
import faiss
from typing import List, Dict
import os
import pickle
import torch

class EmbeddingModel:
    def __init__(self,peft_model_path="models/vietnamese-sbert-lora-med",base_model_path='keepitreal/vietnamese-sbert'):
        # load base model 
        self.base_model = AutoModel.from_pretrained(base_model_path)
        # load adapter LoRA
        lora_model = PeftModel.from_pretrained(self.base_model, peft_model_path)

        # load model
        word_embedding_model = models.Transformer(base_model_path,max_seq_length=256)
        word_embedding_model.auto_model = lora_model
        
        self.pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        
        self.model = SentenceTransformer(modules=[word_embedding_model,self.pooling_model])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.texts = []
        self.faiss_index = None
        
        print("Embedding model loaded successfully!")
        
    
    def _create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        n_samples = embeddings.shape[0]
        nlist = int(np.sqrt(n_samples))
        quantizer = faiss.IndexFlatIP(768)
        index = faiss.IndexIVFFlat(quantizer,768,nlist)
        
        print(f"Training FAISS index with {nlist} clusters...")
        index.train(embeddings.astype('float32'))
        
        return index 
    
    def encode_and_save(self, texts, faiss_path, corpus_path, batch_size = 8, save =True) -> None:
        all_embeddings = []
        for i in range(0,len(texts),batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                device=self.device
            )
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        
        self.texts.extend(texts)
        
        if self.faiss_index is None:
            self.faiss_index = self._create_faiss_index(embeddings)
            self.faiss_index.add(embeddings.astype('float32'))
        else:
            self.faiss_index.add(embeddings.astype('float32'))
        
        if save:
            self.save_index(faiss_path, corpus_path)
        
        print(f"Encoded and added {len(texts)} texts. Total: {len(self.texts)}")
    
    
    def search(self, query, faiss_path, corpus_path, k = 5, score_threshold:float = 0.0) -> List[Dict]:
        if self.faiss_index is None:
            self.load_index(faiss_path,corpus_path)
            
        query_embedding = self.model.encode([query],device=self.device)
        
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'),k)
        
        result = []
        for score, idx in zip(scores[0],indices[0]):
            if idx < len(self.texts) and score > score_threshold:
                result.append({
                    'text' : self.texts[idx],
                    'score': float(score),
                    'index': idx
                })
        
        return result
    def save_index(self, faiss_path, corpus_path):
        os.makedirs(os.path.dirname(faiss_path),exist_ok=True)
        os.makedirs(os.path.dirname(corpus_path),exist_ok=True)
        
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index,faiss_path)
            print(f"Faiss index save to {faiss_path}")
            
        # Save mapping
        with open(corpus_path,'wb') as f:
            pickle.dump(self.texts,f)
        print(f"Text saved to {corpus_path}")
        
    def load_index(self, faiss_path, corpus_path):
        # load faiss index
        if os.path.exists(faiss_path):
            self.faiss_index = faiss.read_index(faiss_path)
            print(f"Faiss index loaded successfully")
        else:
            raise FileNotFoundError(f"Faiss index not found: {faiss_path}")

        # load text mapping
        if os.path.exists(corpus_path):
            with open(corpus_path,'rb') as f:
                corpus = pickle.load(f)
        
            self.texts = corpus
            print(f"Text loaded successfully from {corpus_path}")
        else:
            raise FileNotFoundError(f"Text not found: {corpus_path}")
            

def main():
    embedding_model = EmbeddingModel()
    text = ["Sốt cao ở trẻ cần hạ sốt bằng paracetamol, chườm mát và theo dõi thêm triệu chứng."]
    embedding_model.encode_and_save(texts = text,
                                    faiss_path="data/faiss/faiss_index.faiss",
                                    corpus_path="data/faiss/corpus_RAG.pkl")
    
    
if __name__ == "__main__":
    main()
    
    

