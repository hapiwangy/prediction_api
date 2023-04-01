import requests
import json
url = "http://127.0.0.1:5051/"
print(requests.post(url=url, json = {"sent": "recycle is very important"}).text)