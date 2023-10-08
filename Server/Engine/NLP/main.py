from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTVectorStoreIndex, LLMPredictor, PromptHelper,StorageContext, load_index_from_storage,ServiceContext, set_global_service_context,VectorStoreIndex
from langchain import OpenAI
import sys
import os
from IPython.display import Markdown, display


def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    os.environ["OPENAI_API_KEY"] = "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"

    # setup LLM with max_tokens
    llm = OpenAI(model="text-davinci-003", temperature=0.5, max_tokens=num_outputs)

    # setup global service context
    ctx = ServiceContext.from_defaults(llm=llm, chunk_size=600)
    set_global_service_context(ctx)

    documents = SimpleDirectoryReader(directory_path).load_data()
    print(30)
    index = VectorStoreIndex.from_documents(documents)
    print(40)
    # save the index
    index.storage_context.persist(persist_dir="Directory")
    print(20)
    return index
def ask_ai():
    storage_context = StorageContext.from_defaults(persist_dir='Directory')
    index = load_index_from_storage(storage_context)
    while True:
        query = input("What do you want to ask? ")
        response = index.query(query, response_mode="compact")
        display(Markdown(f"Response: <b>{response.response}</b>"))
construct_index(directory_path="Directory")
ask_ai()
print(10)
