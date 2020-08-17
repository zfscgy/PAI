import os
from Task.TaskScriptMaker import gen_task_path

def str_dict_to_int(str_dict: dict):
    int_dict = dict()
    for key_str in str_dict:
        value = str_dict[key_str]
        int_dict[int(key_str)] = value
    return int_dict
