import os
import json
import shutil

from tools.macros import INFO_DIR, RES_DIR, PROJECT_DIR


def distribute_files(project):
    files = os.listdir(f"{RES_DIR}/{project}")
    for file in files:
        print(f"Putting ----- {file}")
        folder = file[:-4]
        if os.path.exists(f"{INFO_DIR}/{project}/{folder}"):
            shutil.copyfile(f"{RES_DIR}/{project}/{file}", f"{INFO_DIR}/{project}/{folder}/{folder}_constraints.txt")


def read_from_benchmark(project):
    with open(f"{PROJECT_DIR}/benchmark.json", "r") as f:
        data = json.load(f)
    for item in data:
        if item["id"] == project:
            constraints = item["constraints"]
            clas = item["class"]
            func = item["func"]

    if clas == "NA":
        funcname = func
    else:
        funcname = clas
    with open(f"{INFO_DIR}/{project}/{funcname}/{funcname}_constraints.txt", "w") as f:
        f.write("Logical format:" + constraints)