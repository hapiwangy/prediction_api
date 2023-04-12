# prediction_api

# 使用前修改
- classify.py 裡面的 model.pt 成自己模型的路徑
- url

# nlp_project_model
 ## 創建虛擬環境
```
virtualenv virt 
```
 ## 允許腳本運行(powershell)
 ```
Set-ExecutionPolicy Unrestricted -Scope Process
.\virt\Scripts\activate.ps1
```
## 安裝套件(會看到(virt)在前面)
```
pip install -r requirements.txt 
python app.py
 ```
 ## 把pip內容print出並加到requirements.txt 
 ```
 pip freeze
 pip freeze > requirements.txt
 ```
 test