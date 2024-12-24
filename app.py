from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Llama model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with the specific model you want to use
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Test the model
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs.input_ids, max_length=300, temperature=0.7, top_p=0.9)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "Create a persona for a data scientist with goals, needs, and traits."
print(generate_text(prompt))