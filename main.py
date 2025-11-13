"""
Example script for running GPT-2 Small with PyTorch
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    """Load and run GPT-2 Small for text generation"""
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model_name = "gpt2"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    model = model.to(device)
    model.eval()
    
    # Example: text generation
    prompt = "The future of AI is"
    print(f"\nPrompt: {prompt}")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=100,
            num_beams=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}\n")
    
    # Memory info
    if device == "cuda":
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()