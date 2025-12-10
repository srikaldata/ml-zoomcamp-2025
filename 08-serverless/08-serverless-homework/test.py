
import requests

# url of the local lambda runtime API
url = "http://localhost:####/2025-12-31/functions/function/invocations"

# url of the image to be predicted
request = {
    "url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
}

# sending the request and fetching a response 
result = requests.post(url, json=request).json()

print(result)

 