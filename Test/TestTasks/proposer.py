import requests

task_request = {
    "task-name": "task-name",
    "model-name": "nn",
    "train-config": {
        "max_iter": 1234,
        "batch_size": 32, 
        "test_per_rounds": 100,
    },
    "clients": {
        "main": {
            "addr": "127.0.0.1",
            "http_port": 8380,
            "computation_port": 8381
        },
        "crypto-producer": {
            "addr": "127.0.0.1",
            "http_port": 8390,
            "computation_port": 8391
        },
        "data-clients": [
            {
                "addr": "127.0.0.1",
                "http_port": 8080,
                "computation_port": 8081,
                "data-file": "data_file.csv",
                "dim": 30
            },
            {
                "addr": "127.0.0.1",
                "http_port": 8082,
                "computation_port": 8083,
                "data-file": "data_file.csv",
                "dim": 40
            }
        ],
        "label-client": {
            "addr": "127.0.0.1",
            "http_port": 8884,
            "computation_port": 8885,
            "label-file": "label-file.csv",
            "loss": "mse",
            "metrics": "auc_ks"
        }
    }
}

resp = requests.post("http://127.0.0.1:8380/createTask", json=task_request)
print(resp.status_code, resp.text)