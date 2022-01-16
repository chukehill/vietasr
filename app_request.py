import requests

query = {"path": "Segment.wav"}
response = requests.post("http://127.0.0.1:3000/", json=query)
print(response.json())
    