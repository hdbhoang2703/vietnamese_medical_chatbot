from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

class GenerateModel:
    def __init__(self, peft_model_path = "models/vit5-lora-med", base_model_path="VietAI/vit5-large"):
        # Load model base (ViT5-large)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path, 
                                                                trust_remote_code=True, 
                                                                use_safetensors=True,
                                                                low_cpu_mem_usage=True)
        
        # Load adapter QLoRA 
        self.config = PeftConfig.from_pretrained(peft_model_path)
        self.lora_model = PeftModel.from_pretrained(self.base_model, peft_model_path)
        
        # Load tokenizer 
        self.tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lora_model.to(self.device)
        
        print("Generate model loaded successfully!")
    
    def answer(self, prompt,**generate_kwargs):
        if not prompt or not isinstance(prompt, str):
            return "Lỗi: Prompt không hợp lệ."

        try:
            encoded_input = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                padding=True,
                truncation=True
            ).to(self.device)

            outputs = self.lora_model.generate(
                **encoded_input,
                max_length=300,
                min_length=20,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.3,
                no_repeat_ngram_size=4,
                length_penalty=0.8,
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
    Bạn là bác sĩ tư vấn y tế. Dưới đây là thông tin y về bệnh liên quan(nếu không liên quan thì hãy tự trả lời):
    ---------------------
    {context}
    ---------------------
    Câu hỏi của người dùng: {query}
    Chọn lọc thông tin cần thiết, trả lời ngắn gọn và chính xác:
    """.strip()

        return self.answer(prompt, **kwargs)

def main():
    query = "Bạn là 1 chatbot hỗ trợ ý tế. Hãy trả lời: Bệnh tiểu đường có triệu chứng gì"     
    generate_model = GenerateModel()
    reponse = generate_model.answer(query)
    print(reponse)

if __name__ == "__main__":
    main()