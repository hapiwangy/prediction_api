from flask import Flask, jsonify, request
import classify
import torch 
import transformers
import seven
from flask_cors import CORS
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
model = classify.model_predict()
app = Flask(__name__)
cors = CORS(app, resources={r"/": {"origins": "*"}})

@app.route('/', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        model.consequence(data["sent"])
        return jsonify(result=model.answer)
    return "hello world"

@app.route('/banana', methods=['POST', 'GET'])
def predict2():
    if request.method == 'POST':
        data = request.get_json()
        seven.classfy(data["sent"])
        return jsonify(result=seven.cons)
    return "hello world"

if __name__ == '__main__':
    app.run(port=5051, debug = True)