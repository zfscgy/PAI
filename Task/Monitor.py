import os
import pandas as pd
from Task.TaskScriptMaker import gen_task_path
import Task.TaskConfig as Config


def task_status(task_name: str, client_id: int):
    task_path = gen_task_path(task_name, client_id)
    if os.path.isdir(task_path):
        if os.path.isfile(task_path + "finished"):
            return {"status": "ok", "msg": "Finished"}
        elif os.path.isfile(task_path + "run.lock"):
            return {"status": "ok", "msg": "Running"}
        elif os.path.isfile(task_path + "failed"):
            return {"status": "ok", "msg": "Failed"}
        else:
            return {"status": "ok", "msg": "Created"}
    else:
        return {"status": "ok", "msg": "NotExist"}


def data_status(data_name: str):
    data_path = Config.DataPath + data_name + "/train.csv"
    data_test_path = Config.DataPath + data_name + "/test.csv"
    if not os.path.isfile(data_path):
        return {"status": "err", "msg": "NotExist"}
    else:
        try:
            shape_train = pd.read_csv(data_path, header=None).shape
            shape_test = pd.read_csv(data_test_path, header=None).shape
            return {"status": "ok", "msg": [shape_train[0], shape_test[0]]}
        except Exception as e:
            return {"status": "err", "msg": str(e)}


def train_result(task_name: str, client_id: int):
    task_path = gen_task_path(task_name, client_id)
    if not os.path.isdir(task_path):
        return {"status": "err", "msg": "NotExist"}
    else:
        try:
            record = pd.read_csv(task_path + "record.csv", header=None).values.tolist()
            return {"status": "ok", "msg": record}
        except Exception as e:
            return {"status": "err", "msg": str(e)}

