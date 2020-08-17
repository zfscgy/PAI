import requests
# resp = requests.get("http://127.0.0.1:8380/startTask", params={"task_name": "test-datagen"})
resp = requests.get("http://10.214.192.22:8380/startTask", params={"task_name": "test-datagen"})
print(resp.status_code, resp.text)