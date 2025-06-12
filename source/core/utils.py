import os
from typing import List
import numpy 

def load_corpus(folder_path, encoding = "utf-8") -> List[str]:
    corpus = []
    for filename in os.listdir(folder_path):
        if filename.endswith("txt"):
            file_path = os.path.join(folder_path,filename)
            try:
                with open(file_path,"r",encoding=encoding) as f:
                    text = f.read().strip()
                    if text:
                        corpus.append(text)
            except Exception as e:
                print(f"File read failed : {file_path} {e}")
    
    print(f"Load corpus successfully : {folder_path}")
    return corpus

def cleaning_corpus(corpus: List[str]) -> List[List[str]]:
    clean_corpus = []
    for text in corpus:
        clean_text = [sentence.strip() for sentence in text.split("\n") if sentence.strip()]
        clean_corpus.append(clean_text)
    return clean_corpus

def chunking_corpus(corpus: List[List[str]], chunk_size: int) -> List[str]:
    chunk_corpus = []
    idx = 0
    length = len(corpus)

    while idx < length:
        current_doc = corpus[idx]
        total_words = sum(len(sentence.split()) for sentence in current_doc)

        if total_words < chunk_size:
            combined_sentences = current_doc.copy()
            idx += 1
            while idx < length and \
                (sum(len(sentence.split()) for sentence in combined_sentences) +
                 sum(len(sentence.split()) for sentence in corpus[idx])) <= chunk_size:
                combined_sentences.extend(corpus[idx])
                idx += 1
            chunk_corpus.append(".".join(combined_sentences))
        else:
            flat_sentences = current_doc
            sub_chunk = []
            word_count = 0
            for sentence in flat_sentences:
                sentence_word_count = len(sentence.split())
                if word_count + sentence_word_count <= chunk_size:
                    sub_chunk.append(sentence)
                    word_count += sentence_word_count
                else:
                    chunk_corpus.append(".".join(sub_chunk))
                    sub_chunk = [sentence]
                    word_count = sentence_word_count
            if sub_chunk:
                chunk_corpus.append(".".join(sub_chunk))
            idx += 1

    return chunk_corpus
