import torch
import transformers
from transformers import BertTokenizer
import numpy as np
import sys

labels = ["E","S","G"]
cons=""
def classfy(sentence:str):
    device = torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
    model = torch.load(r"C:\Users\user\Downloads\ESGsingle.pth", map_location=torch.device('cpu'))
    model.eval()
    model.to(device)
    encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt',
        )
    output = model(encoded_dict['input_ids'], token_type_ids=None)
    logits = output[0]
    logits = logits.detach().cpu().numpy()
    flat_prediction = np.argmax(logits, axis=1).flatten()
    cons=labels[flat_prediction[0]]
    return labels[flat_prediction[0]]