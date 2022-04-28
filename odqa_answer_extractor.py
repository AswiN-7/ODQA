from transformers import  pipeline

# model_name = "deepset/xlm-roberta-large-squad2"
model_name = "AswiN037/xlm-roberta-squad-tamil"

answer_extract = pipeline('question-answering', model=model_name, tokenizer=model_name)


