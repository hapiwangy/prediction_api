import torch
import transformers
from transformers import BertTokenizer
import numpy as np
import sys
labels = ["Climate","Natural capital","Pollution&Waste","Env. Opportunities","Human Capital","Product Liability","Social Opportunities","Corporator Governance","Corporator Behavior"]
device = 'cpu' 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
MAX_LEN = 256
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', outputs_attentions=True)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 9)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
class model_predict:
    class BERTClass(torch.nn.Module):
        def __init__(self):
            super(BERTClass, self).__init__()
            self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', outputs_attentions=True)
            self.l2 = torch.nn.Dropout(0.3)
            self.l3 = torch.nn.Linear(768, 9)
        
        def forward(self, ids, mask, token_type_ids):
            _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
            output_2 = self.l2(output_1)
            output = self.l3(output_2)
            return output
    # 初始化模型相關數據
    def __init__(self):
        self.model = torch.load(r"C:\Users\user\Downloads\model_third.pt", map_location='cpu')
        self.tokenizer = tokenizer
        self.device = device
        self.labels = labels
    # 轉換輸出並回傳相關數據
    def change_input(self, input: str):
        after_encode_inputs = tokenizer.encode_plus(
            input,
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding=True,
            return_token_type_ids=True,
            truncation=True
        )
        self.ids = torch.tensor([after_encode_inputs['input_ids']], dtype=torch.long)
        self.mask = torch.tensor([after_encode_inputs['attention_mask']], dtype=torch.long)
        self.token_type_ids = torch.tensor([after_encode_inputs["token_type_ids"]], dtype=torch.long)
    # 處理數據、得到答案
    def consequence(self, input:str):
        self.change_input(input)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.ids, self.mask, self.token_type_ids)
        outputs = np.array(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        outputs = outputs >= 0.048
        outputs = outputs[0]
        ans = []
        for index , x in enumerate(outputs):
            if x == True:
                ans.append(labels[index])
        ans = ",".join(ans)
        if len(ans) == 0:
            self.answer = f"this sentence is unrelated!!"
        else:
            self.answer = f"this sentence's labels include {ans}"
if __name__ == "__main__":
    model = model_predict()
    model.consequence("government is very important!!")
    print(model.answer)