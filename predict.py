# predict.py
import os
import sys
import time
from cog import BasePredictor, Input

# Import modules từ source
sys.path.append('./source')
from source.core.rag_pipeline import RAGpipeline
from source.core.embedding_model import EmbeddingModel

class Predictor(BasePredictor):
    def setup(self) -> None:
        print("Setting up RAG pipeline...")
        
        # Initialize RAG pipeline
        self.rag = RAGpipeline()
        
        # Setup paths
        self.faiss_path = "data/faiss/faiss_index.faiss"
        self.corpus_path = "data/Corpus_RAG"
        
        if not os.path.exists(self.faiss_path):
            print("FAISS index not found. Creating embeddings...")
            os.makedirs("data/faiss", exist_ok=True)
            self.rag.embedding_data(self.corpus_path, 200)
            print("FAISS index created successfully!")
        else:
            print("FAISS index found. Loading...")
        
        print("RAG pipeline setup completed!")

    def predict(
        self,
        query: str = Input(
            description="Nhập câu hỏi y tế của bạn",
            default="Triệu chứng của bệnh tiểu đường là gì?"
        ),
        top_k: int = Input(
            description="Số lượng documents liên quan để tham khảo",
            default=3,
            ge=1,
            le=10
        )
    ) -> str:
        """
        Trả lời câu hỏi y tế bằng RAG pipeline
        """
        
        if not query.strip():
            return "Lỗi: Vui lòng nhập câu hỏi."
        
        try:
            start_time = time.time()
            
            # Chạy RAG pipeline
            answer = self.rag.run(query, top_k)
            
            end_time = time.time()
            processing_time = round(end_time - start_time, 2)
            
            # Trả về kết quả với thời gian xử lý
            result = f"{answer}\n\n⏱️ Thời gian xử lý: {processing_time}s"
            
            return result
            
        except Exception as e:
            error_msg = f"Lỗi khi xử lý câu hỏi: {str(e)}"
            print(f"Error in prediction: {e}")
            return error_msg