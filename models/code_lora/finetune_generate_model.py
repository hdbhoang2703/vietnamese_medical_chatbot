from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
import torch
from datasets import load_dataset
from datasets import DatasetDict
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import get_peft_model, LoraConfig, TaskType

# load model
model_id = "VietAI/vit5-base"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16  
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# use QLoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
    target_modules=["q", "v"], 
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# tokenizer train_dataset and processing 
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["question"], max_length=512, truncation=True, padding=True
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["answer"], max_length=256, truncation=True, padding=True
        )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

raw_dataset = load_dataset("json",data_files = "/kaggle/input/data-fine-tune/data_train.json" , split = "train")
dataset = raw_dataset.train_test_split(test_size=0.1, seed=2025)

dataset_dict = DatasetDict({
    'train': dataset['train'],
    'validation': dataset['test']
})

tokenized_datasets = dataset_dict.map(
    preprocess_function,
    batched=True,
    remove_columns=["question", "answer"],
    num_proc=1
)

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

# load training config
training_args = Seq2SeqTrainingArguments(
    output_dir="./output-vit5-qlora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=4,
    learning_rate=2e-4,
    fp16=True,
    bf16=False, 
    logging_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=400, 
    report_to="none",
    remove_unused_columns=False
)

# train model
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

trainer.train(resume_from_checkpoint=True)


# save fine-tune model
model.save_pretrained("./ViT5_finetuned")
tokenizer.save_pretrained("./ViT5_finetuned")

