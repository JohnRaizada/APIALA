print("Initializing...")
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
model_name = "deepset/roberta-base-squad2"
print("Warming Up...")
# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
print("Almost There...")
# Load the text file with utf-8 encoding
with open('context.txt', 'r', encoding='utf-8') as file:
    context = file.read()
print("Ready...")
# Start the chat loop
while True:
    # Get user input
    prompt = input(">>>>>>    ")
    # Check if the conversation should be ended
    if prompt.lower() == "quit":
        print("Goodbye...")
        break
    QA_input = {
    'question': prompt,
    'context': context
    }
    print("Generating Answers....")
    res = nlp(QA_input)
    print("APIALA:  " + res['answer'])
