# from transformers import pipeline, AutoTokenizer

# # model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# classifier = pipeline("sentiment-analysis")
# print(classifier("I've been waiting for a HuggingFace course my whole life."))

# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# sequence = "Using a Transformer network is simple"
# tokens = tokenizer.tokenize(sequence)
# print(tokens)

# ids = tokenizer.convert_tokens_to_ids(tokens)

# print(ids)

# decoded_string = tokenizer.decode([7993, 170, 13809, 23763, 2443, 1110, 3014])
# print(decoded_string)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
print(output)