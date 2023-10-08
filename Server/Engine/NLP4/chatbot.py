from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Initialize the model and tokenizer
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load the text file with utf-8 encoding
with open('context.txt', 'r', encoding='utf-8') as file:
    context = file.read()

# Tokenize the context and truncate it to the tokenizer's max length
context_ids = tokenizer.encode(context, return_tensors='pt', truncation=True, max_length=tokenizer.model_max_length)

# Start the chat loop
while True:
    # Get user input
    text = input(">> You: ")

    # Check if the conversation should be ended
    if text.lower() == "quit":
        break

    # Encode the user input and add end of string token
    inputs = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")

    # Concatenate new user input with context (if there is)
    bot_input_ids = torch.cat([context_ids, inputs], dim=-1) if context_ids.shape[0] > 0 else inputs

    # Generate a response
    outputs = model.generate(bot_input_ids, max_length=1000, do_sample=True, top_p=0.95, top_k=0, temperature=0.75)

    # Decode the response
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"FLAN-T5: {output}")
