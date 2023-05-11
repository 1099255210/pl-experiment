import requests
import json

url = "http://127.0.0.1:5000/u2net"

payload = json.dumps({
  "path": "C:\\Users\\hg\\Documents\\projects\\pl-experiment\\u2net\\test_assets\\pixelmator.jpg"
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)