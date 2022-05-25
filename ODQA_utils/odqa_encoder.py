import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoModel, AutoTokenizer

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("AswiN037/sentence-t-roberta-large-wechsel-tamil")
print("Retriever model loaded")


def encode(text):
    embeddings = model.encode(text)
    return [embeddings.tolist()]

# tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
# model = AutoModel.from_pretrained('ai4bharat/indic-bert')

# import torch
# import torch.nn.functional as F

# #Mean Pooling - Take average of all tokens
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
# #Encode text
# def encode(texts):
#     # Tokenize sentences
#     doc_stride = 128
#     encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512, stride=doc_stride, return_overflowing_tokens = True)
#     encoded_input.pop("overflow_to_sample_mapping")

#     # Compute token embeddings
#     with torch.no_grad():
#         model_output = model(**encoded_input, return_dict=True)

#     # Perform pooling
#     embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

#     # Normalize embeddings
#     embeddings = F.normalize(embeddings, p=2, dim=1)
    
#     return embeddings.tolist()
