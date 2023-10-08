from transformers import GPT2LMHeadModel, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
print("Welcome to the future of APIALA")
while True:
    input_text = input(">>>>>>   ")
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=50)
    output_text = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("APIALA:  "+output_text)
