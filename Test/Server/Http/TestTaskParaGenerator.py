import json

from Server.HttpServer.TaskParaGenerator import TaskParaGenerator, generate_paras

post_json = {
    "task_name": "test-task",
    "model_name": "shared_nn",
    "configs": {
        "train_config": {
            "max_iter": 1234,
            "loss_func": "mse",
            "metrics": "auc_ks"
        }
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
        "feature_clients": [
            {
                "addr": "127.0.0.1",
                "http_port": 8084,
                "computation_port": 8085,
                "data_file": "Splitted_Indexed_Data/credit_default_data1.csv",
                "dim": 30
            },
            {
                "addr": "127.0.0.1",
                "http_port": 8082,
                "computation_port": 8083,
                "data_file": "Splitted_Indexed_Data/credit_default_data2.csv",
                "dim": 40
            }
        ],
        "label_client": {
            "addr": "127.0.0.1",
            "http_port": 8884,
            "computation_port": 8885,
            "data_file": "Splitted_Indexed_Data/credit_default_label.csv",
            "dim": 1,
        }
    }
}

def test_common_paras():
    para_gen = TaskParaGenerator(post_json)
    err = para_gen.generate_paras()
    print("Generate paras: \n", err)
    print(json.dumps(para_gen.client_paras, indent=2))

def test_shared_mpc_paras():
    gen = generate_paras(post_json)
    print(gen)
    print(json.dumps(gen.client_paras, indent=2))

if __name__ == '__main__':
    print("========= Test generate common mpc paras ==========")
    test_common_paras()
    print("========= Test generate shared mpc paras ==========")
    test_shared_mpc_paras()