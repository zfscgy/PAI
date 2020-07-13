import os
import pathlib
import shutil
print("Start cleaning task test files...")
for log_file in pathlib.Path("Test/Log/").iterdir():
    os.remove(str(log_file))
for task_json in pathlib.Path("Test/TestTasksRoot/MainServerRoot/").iterdir():
    os.remove(str(task_json))
for task_dir in pathlib.Path("Test/TestTasksRoot/ClientTaskRoot/").iterdir():
    shutil.rmtree(str(task_dir))
print("done.")