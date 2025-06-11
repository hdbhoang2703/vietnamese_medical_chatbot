from .embedding_model import EmbeddingModel
from .generate_model import GenerateModel
from .utils import load_corpus, cleaning_corpus,chunking_corpus

class RAGpipline:
    def __init__(self,
                 faiss_path = "data/faiss/faiss_index.faiss",
                 corpus_path = "data/faiss/corpus_RAG.pkl",
                 ):
        self.embedding_model = EmbeddingModel()
        self.generate_model = GenerateModel()
        
        self.faiss_path = faiss_path
        self.corpus_path = corpus_path 
    
    def embedding_data(self,data_folder,chunk_size):
        raw_data = load_corpus(data_folder)
        clean_data = cleaning_corpus(raw_data)
        data = chunking_corpus(corpus = clean_data,chunk_size = chunk_size)
        
        self.embedding_model.encode_and_save(data,
                                             self.faiss_path,
                                             self.corpus_path,
                                             batch_size = 8
                                             )
        print("Embedding and save data successfully")
    def run(self, query, k: int = 5):
        # encode query and search context
        retrieved_docs = self.embedding_model.search(
                                                query = query,
                                                faiss_path=self.faiss_path,
                                                corpus_path=self.corpus_path,
                                                k = k)
        
        context = "\n".join([doc['text'] for doc in retrieved_docs])
        # generate answer
        response = self.generate_model.generate_from_context(context, query)
        
        return response
    
def main():
    rag = RAGpipline()
    rag.embedding_data("data/Corpus_RAG",200)
    query = "Tôi bị tiêu chảy 2 ngày rồi, có cần uống Oresol không?"
    answer = rag.run(query)
    print("Câu trả lời:")
    print(answer)


if __name__ == "__main__":
    main()