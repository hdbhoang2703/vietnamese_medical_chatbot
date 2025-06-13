---
base_model: VietAI/vit5-large
library_name: peft
---

# Model Card for Vietnamese ViT5 Generator (Fine-tuned with QLoRA 4-bit)

Đây là mô hình sinh văn bản tiếng Việt, được fine-tune từ **VietAI/vit5-large** bằng phương pháp **QLoRA (4-bit quantization)** thông qua thư viện **PEFT**. Mô hình đóng vai trò "Generator" trong kiến trúc **RAG (Retrieval-Augmented Generation)** dành cho chatbot y tế tiếng Việt.

## Model Details

### Model Description

- **Tên mô hình**: ViT5-QLoRA-Generator (Vietnamese T5 Large, fine-tuned)
- **Ứng dụng chính**: Sinh câu trả lời từ câu hỏi và tài liệu y khoa (QA + Generation).
- **Mô hình gốc**: [VietAI/vit5-large](https://huggingface.co/VietAI/vit5-large)
- **Phương pháp Fine-tune**: QLoRA (4-bit quantized LoRA)
- **Thư viện sử dụng**: PEFT, Transformers, Datasets, BitsAndBytes
- **Ngôn ngữ**: Tiếng Việt
- **License**: Apache 2.0 
- **Loại mô hình**: Seq2Seq Text-to-Text Generation
- **Fine-tuned từ**: VietAI/vit5-large

## Uses

### Direct Use

- Sinh câu trả lời cho câu hỏi, dựa trên ngữ cảnh tài liệu đầu vào (passage + question).
- Có thể ứng dụng trong chatbot, trợ lý ảo, hoặc hệ thống hỏi đáp tiếng Việt.

### Downstream Use

- Kết hợp với retriever như SBERT trong kiến trúc RAG để xây dựng chatbot y tế thông minh.

### Out-of-Scope Use

- Không thích hợp để trả lời các câu hỏi không có ngữ cảnh.
- Không phù hợp cho sinh văn bản đa ngôn ngữ (chỉ hỗ trợ tiếng Việt).

## Bias, Risks, and Limitations

- Có thể tạo ra câu trả lời sai lệch nếu tài liệu đầu vào không chính xác.
- Không đảm bảo tính chính xác tuyệt đối trong lĩnh vực y tế.
- Mô hình không có khả năng từ chối trả lời nếu không chắc chắn.

## How to Get Started with the Model

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# Cấu hình BitsAndBytes cho 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load mô hình gốc (ViT5-large) với 4-bit
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    "VietAI/vit5-large",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    use_safetensors=True,
    low_cpu_mem_usage=True
)

# Load adapter QLoRA
model = PeftModel.from_pretrained(base_model, "path/to/your-qlora-model")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("path/to/your-qlora-model")

# Tạo input và sinh câu trả lời
input_text = "question: Tôi bị sốt và đau họng. Tôi nên làm gì? context: Sốt và đau họng có thể là dấu hiệu của viêm họng do virus..."
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
