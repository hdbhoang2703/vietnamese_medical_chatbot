from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import TripletEvaluator


# split dataset
raw_dataset = load_dataset("json",data_files = "/kaggle/input/emb-data/data_finetune_emb_model.json" , split = "train")
raw_dataset.remove_columns('negative_type')

train_val = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_val_dataset = train_val['train']
test_dataset = train_val['test'] 

train_val_split = train_val_dataset.train_test_split(test_size=0.1111, seed=42)
train_dataset = train_val_split['train']
val_dataset = train_val_split['test'] 

# load model
model_name = "keepitreal/vietnamese-sbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)

# LoRA config
lora_config = LoraConfig(
    r=16,                    
    lora_alpha=32,           
    target_modules=["query", "key", "value", "dense"],  # Full attention
    lora_dropout=0.1,        
    bias="none",             
    task_type="FEATURE_EXTRACTION"  
)

# Gắn model LoRA vào sentence-transformers
lora_model = get_peft_model(base_model, lora_config)
word_embedding_model = models.Transformer(model_name, max_seq_length=256)
word_embedding_model.auto_model = lora_model  

# create SentenceTransformer model
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# preprocessing dataset (tokenizer,...)
train_samples = [InputExample(texts=[d["query"], d["positive"], d["negative"]]) for d in train_dataset]
val_samples   = [InputExample(texts=[d["query"], d["positive"], d["negative"]]) for d in val_dataset]
train_loader = DataLoader(train_samples, shuffle=True, batch_size=8)

# set loss function
train_loss = losses.TripletLoss(model=model)

# evaluate
evaluator = TripletEvaluator.from_input_examples(val_samples, name="val-eval")

# train 
model.fit(
    train_objectives=[(train_loader, train_loss)],
    evaluator=evaluator,
    epochs=3,
    warmup_steps=100,
    evaluation_steps=500, 
    save_best_model=True,
    show_progress_bar=True,
    output_path="./lora-vietnamese-sbert-triplet"
)