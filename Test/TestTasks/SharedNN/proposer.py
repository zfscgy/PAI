import requests


task_request = {
    "task_name": "test-shared_nn",
    "clients": [
        {
            "role": "main_client",
            "addr": "127.0.0.1",
            "http_port": 8377,
            "client_config": {
                "client_type": "shared_nn_main",
                "computation_port": 8378,
                "in_dim": 1,
                "out_dim": 1,
                "layers": [],
                "test_per_batches": 101,
                "max_iter":12345
            }
        },
        {
            "role": "crypto_producer",
            "addr": "127.0.0.1",
            "http_port": 6666,
            "client_config": {
                "client_type": "triplet_producer",
                "computation_port": 6699,
                "listen_clients": [2, 3]
            }
        },
        {
            "role": "feature_client",
            "addr": "127.0.0.1",
            "http_port": 8084,
            "client_config": {
                "client_type": "shared_nn_feature",
                "computation_port": 8085,
                "data_path": "test-f1",
            }
        },
        {
            "role": "feature_client",
            "addr": "127.0.0.1",
            "http_port": 8082,
            "client_config": {
                "computation_port": 8083,
                "client_type": "shared_nn_feature",
                "data_path": "test-f2"
            }
        },
        {
            "role": "label_client",
            "addr": "127.0.0.1",
            "http_port": 8884,
            "client_config": {
                "computation_port": 8885,
                "client_type": "shared_nn_label",
                "data_path": "test-l",
                "loss": "mse",
                "metric": "metrics_pack1"
            }
        }
    ]
}

resp = requests.post("http://127.0.0.1:8380/createTask", json=task_request)
print(resp.status_code, resp.text)
