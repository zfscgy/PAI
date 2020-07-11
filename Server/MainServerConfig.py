class MainConfig:
    server_log_path = "Test/Log/"
    server_task_root = "Test/TestTasks/MainServer/"

    client_protocol = "http://"
    #
    learning_rate = 0.1
    batch_size = 64
    learn_config = {
        "loss": "mse",
        "metrics": "auc_ks"
    }