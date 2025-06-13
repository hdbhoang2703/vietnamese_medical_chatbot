import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

class GenerateModel:
    def __init__(self, 
                 peft_model_path="models/vit5-lora-med", 
                 base_model_path="VietAI/vit5-base"): 
        # BitsAndBytes config 
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load base model
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )

        # Load adapter LoRA
        self.lora_model = PeftModel.from_pretrained(self.base_model, peft_model_path)

        # Load tokenizer từ adapter
        self.tokenizer = AutoTokenizer.from_pretrained(peft_model_path)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lora_model.to(self.device)

        print("Generate model loaded successfully!")

    def answer(self, prompt, **generate_kwargs):
        if not prompt or not isinstance(prompt, str):
            return "Lỗi: Prompt không hợp lệ."

        try:
            encoded_input = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                padding=True,
                truncation=True
            )
            
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            outputs = self.lora_model.generate(
                **encoded_input,
                max_length=400,
                min_length=20,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **generate_kwargs
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.strip()

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return "Lỗi: Hết bộ nhớ GPU khi sinh văn bản."
        except Exception as e:
            return f"Lỗi trong quá trình generate: {str(e)}"

    def generate_from_context(self, context: str, query: str, **kwargs):
        prompt = f"""
Bạn là bác sĩ tư vấn y tế. Dưới đây là thông tin y về bệnh liên quan (nếu không liên quan thì hãy tự trả lời):
---------------------
{context}
---------------------
Câu hỏi của người dùng: {query}
Chọn lọc thông tin cần thiết, trả lời ngắn gọn và chính xác:
""".strip()

        return self.answer(prompt, **kwargs)

def main():
    query = "Bạn là 1 chatbot hỗ trợ y tế. Hãy trả lời: Bệnh tiểu đường có triệu chứng gì?"     
    generate_model = GenerateModel()
    response = generate_model.answer(query)
    print("Trả lời:\n", response)

if __name__ == "__main__":
    main()
