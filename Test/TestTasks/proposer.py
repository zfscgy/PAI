import requests

task_request = {
    "task_name": "test-task",
    "model-name": "nn",
    "train-config": {
        "max_iter": 1234,
        "batch_size": 32, 
        "test_per_rounds": 100,
    },
    "clients": {
        "main_client": {
            "addr": "127.0.0.1",
            "http_port": 8377,
            "computation_port": 8378
        },
        "crypto_producer": {
            "addr": "127.0.0.1",
            "http_port": 8390,
            "computation_port": 8391
        },
        "data_clients": [
            {
                "addr": "127.0.0.1",
                "http_port": 8084,
                "computation_port": 8085,
                "data-file": "Splitted_Indexed_Data/credit_default_data1.csv",
                "dim": 30
            },
            {
                "addr": "127.0.0.1",
                "http_port": 8082,
                "computation_port": 8083,
                "data-file": "Splitted_Indexed_Data/credit_default_data2.csv",
                "dim": 40
            }
        ],
        "label_client": {
            "addr": "127.0.0.1",
            "http_port": 8884,
            "computation_port": 8885,
            "data-file": "Splitted_Indexed_Data/credit_default_data3.csv",
            "loss": "mse",
            "metrics": "auc_ks"
        }
    }
}

resp = requests.post("http://127.0.0.1:8380/createTask", json=task_request)
print(resp.status_code, resp.text)