import requests
resp = requests.get("http://127.0.0.1:8380/startTask", params={"task_name": "test-shared_nn"})
print(resp.status_code, resp.text)
