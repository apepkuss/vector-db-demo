
from langchain.document_loaders import PyMuPDFLoader
import nltk.data
# nltk.download("punkt")
import openai

loader = PyMuPDFLoader("paper.pdf")
pages = loader.load()
print(f"num of pages: {len(pages)}")
# print(f"{pages[0].page_content}")


text = pages[0].page_content

# text = "Hello everyone. Welcome to GeeksforGeeks. You are studying NLP article"

# Loading PunktSentenceTokenizer using English pickle file
tokenizer = nltk.data.load("tokenizers/punkt/PY3/english.pickle")

lines = tokenizer.tokenize(text)

print(f"len of tokens: {len(lines)}")
# print(f"line0: {lines[0]}")
# print(f"line1: {lines[1]}")
# print(f"line2: {lines[2]}")

input = [lines[1], lines[2]]
print(f"input: {input}")

# def get_embedding(input: list[str], engine="text-embedding-ada-002") -> list[float]:
#     sentences = []
#     for s in input:
#         text = s.replace("\n", " ")
#         sentences.append(text)

#     embeddings = []
#     for d in openai.Embedding.create(input=sentences, engine=engine)["data"]:
#         embedding.append(d["embedding"])
#     return embeddings

# embedding = get_embedding(input)
