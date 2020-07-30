import os, re, json
import Task.TaskConfig as Config


def gen_task_path(task_name: str, client_id: int):
    return Config.TaskRootPath + task_name + '-' + str(client_id) + "/"


def create_task_pyscript(**kwargs):
    task_name = kwargs["task_name"]
    client_id = kwargs["client_id"]
    if not os.path.isdir(Config.TaskRootPath):
        os.mkdir(Config.TaskRootPath)
    task_path = gen_task_path(task_name, client_id)
    if not os.path.isdir(task_path):
        os.mkdir(task_path)
    client_config = kwargs["client_config"]

    if "raw_data_path" in client_config:
        client_config["raw_data_path"] = Config.RawDataPath + client_config["raw_data_path"]
    if "data_path" in client_config:
        client_config["data_path"] = Config.DataPath + client_config["data_path"] + "/"
    if "out_data_path" in client_config:
        if not os.path.isdir(Config.DataPath):
            os.mkdir(Config.DataPath)
        if not os.path.isdir(Config.DataPath + client_config["out_data_path"]):
            os.mkdir(Config.DataPath + client_config["out_data_path"])
        client_config["out_data_path"] = Config.DataPath + client_config["out_data_path"] + "/"

    task_script_string = \
        "import sys\n" +\
        "import os\n" +\
        "sys.path.append('{}')\n".format(re.escape(os.getcwd())) +\
        "from Task.Task import Task\n" + \
        "null = None\n" +\
        "task_config = {}\n".format(json.dumps(kwargs, indent=4)) +\
        "task = Task(**task_config)\n" +\
        "task.start()\n"
    pyscripte_file = open(task_path + "task.py", "w+")
    pyscripte_file.write(task_script_string)
    pyscripte_file.close()
