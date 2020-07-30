import os
from Task.TaskScriptMaker import gen_task_path

def str_dict_to_int(str_dict: dict):
    int_dict = dict()
    for key_str in str_dict:
        value = str_dict[key_str]
        int_dict[int(key_str)] = value
    return int_dict


def task_status(task_name: str, client_id: int):
    task_path = gen_task_path(task_name, client_id)
    if os.path.isdir(task_path):
        if os.path.isfile(task_path + "finished"):
            return "Finished"
        elif os.path.isfile(task_path + "run.lock"):
            return "Running"
        elif os.path.isfile(task_path + "failed"):
            return "Failed"
        else:
            return "Created"
    else:
        return "Not Exist"
