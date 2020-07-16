import os, re, json
from Server.TaskControl.MPCTask import MPCModels, MPCRoles
import Server.TaskControl.TaskConfig as Config


def create_task_from_paras(task_paras):
    model_name = task_paras["model_name"]
    role = task_paras["role"]
    if model_name in [MPCModels.SharedLR, MPCModels.SharedNN]:
        if role in [MPCRoles.FeatureProvider, MPCRoles.LabelProvider]:
            from Server.TaskControl.SharedNNTasks import DataProviderTask
            return DataProviderTask(**task_paras)
        elif role is MPCRoles.CryptoProvider:
            from Server.TaskControl.TripletProducerTask import TripletProducerTask
            return TripletProducerTask(**task_paras)
        elif role is MPCRoles.MainProvider:
            from Server.TaskControl.SharedNNMainTask import MainProviderTask
            return MainProviderTask(**task_paras)
        else:
            # This code shall never be reached
            return None


def create_task_pyscript(**kwargs):
    os.mkdir(Config.TaskRootPath + kwargs["task_name"] + "-%d" % (kwargs["client_id"]))
    task_script_string = \
        "import sys\n" +\
        "sys.path.append('{}')\n".format(re.escape(os.getcwd())) +\
        "from Server.TaskControl.TaskScriptMaker import create_task_from_paras\n" + \
        "null = None\n" +\
        "task_config = {}\n".format(json.dumps(kwargs, indent=4)) +\
        "task = create_task_from_paras(task_config)\n" +\
        "task.start()\n"
    pyscripte_file = open(Config.TaskRootPath + kwargs["task_name"] + "-%d" % kwargs["client_id"] + "/task.py", "w+")
    pyscripte_file.write(task_script_string)
    pyscripte_file.close()
