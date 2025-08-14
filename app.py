# app.py
from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import re # <-- Added import for regex

app = Flask(__name__)

# --- Model Loading (on startup) ---
print("Loading TinyLlama MedBot model...")
hf_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_path = "./medbot_tinyllama_final_model"

# Load base model and tokenizer (CPU only)
model = AutoModelForCausalLM.from_pretrained(
    hf_model,
    low_cpu_mem_usage=True,
    device_map="cpu", # Explicitly load on CPU
    torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(hf_model)
tokenizer.pad_token = tokenizer.eos_token

# Load LoRA adapters (your fine-tuned weights)
model = PeftModel.from_pretrained(model, model_path)
model.eval() # Set to evaluation mode
print("Model loaded successfully!")

# --- Add this function ---
def trim_to_last_sentence(text):
    """Trims the text to the last complete sentence ending with '.', '?', or '!'."""
    # Find all sentences ending with ., ?, !
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # If there's only one or no sentence-ending punctuation, return the whole text
    if len(sentences) <= 1:
        return text.strip()
    # Join all sentences except the last one (which might be incomplete)
    trimmed_text = ' '.join(sentences[:-1])
    # Ensure it ends with punctuation
    if trimmed_text and trimmed_text[-1] not in '.!?':
        # If the joined text doesn't end with punctuation, find the last one
        last_punct_pos = max(
            trimmed_text.rfind('.'),
            trimmed_text.rfind('?'),
            trimmed_text.rfind('!')
        )
        if last_punct_pos != -1:
            trimmed_text = trimmed_text[:last_punct_pos + 1]
    return trimmed_text.strip()
# --- End of added function ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        question = request.form['question']
        prompt = f"### Instruction:\nProvide a detailed explanation.\n\n### Input:\n{question}\n\n### Response:\n"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        # inputs are on CPU
        
        # Generate response (adjust parameters as needed)
        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.6,
                top_p=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text.split("### Response:\n")[-1].strip()
        clean_response = trim_to_last_sentence(response) # <-- This line was already using the function
        
        return render_template('result.html', question=question, answer=clean_response)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template('result.html', question=question, answer=error_message)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) # debug=False for production