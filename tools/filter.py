import os
import shutil
import json

from tools.macros import INFO_DIR

def filter_params_and_attributes(project):
    info_path = os.path.join(INFO_DIR, project)
    if not os.path.exists(info_path):
        print(f"Do not have project: {project}, please extract info first")
    folders = os.listdir(info_path)

    count_before = count_directories(info_path)
    for folder in folders:
        if "DS_Store" in folder:
            continue
        
        if os.path.exists(f"{INFO_DIR}/{project}/{folder}/{folder}_pa.json"):
            pa_path = f"{INFO_DIR}/{project}/{folder}/{folder}_pa.json"
            pa_graph = generate_graph(pa_path)
            if not has_cross_parameters(pa_graph):
                os.remove(pa_path)
        if os.path.exists(f"{INFO_DIR}/{project}/{folder}/memberfunc"):
            memberfunc_folders = os.listdir(f"{INFO_DIR}/{project}/{folder}/memberfunc")
            for memberfunc in memberfunc_folders:
                if "DS_Store" in memberfunc:
                    continue
                if os.path.exists(f"{INFO_DIR}/{project}/{folder}/memberfunc/{memberfunc}/{memberfunc}_pa.json"):
                    pa_path = f"{INFO_DIR}/{project}/{folder}/memberfunc/{memberfunc}/{memberfunc}_pa.json"
                    pa_graph = generate_graph(pa_path)
                    if not has_cross_parameters(pa_graph):
                        os.remove(pa_path)

    for folder in folders:
        if not os.path.exists(f"{INFO_DIR}/{project}/{folder}/{folder}_pa.json"):
            sub_pa = False
            if os.path.exists(f"{INFO_DIR}/{project}/{folder}/memberfunc"):
                memberfunc_folders = os.listdir(f"{INFO_DIR}/{project}/{folder}/memberfunc")
                for memberfunc in memberfunc_folders:
                    if "DS_Store" in memberfunc:
                        continue
                    if os.path.exists(f"{INFO_DIR}/{project}/{folder}/memberfunc/{memberfunc}/{memberfunc}_pa.json"):
                        sub_pa = True
                    else:
                        cur_folder = f"{INFO_DIR}/{project}/{folder}/memberfunc/{memberfunc}"
                        shutil.rmtree(cur_folder)
            if not sub_pa:
                cur_folder = f"{INFO_DIR}/{project}/{folder}"
                shutil.rmtree(cur_folder)

    fs = os.listdir(info_path)
    for f in fs:
        if "DS_Store" in folder:
            continue
        if not os.path.exists(f"{INFO_DIR}/{project}/{f}/{f}_pa.json"):
            cur_folder = f"{INFO_DIR}/{project}/{f}"
            shutil.rmtree(cur_folder)

    count_after = count_directories(info_path)
    print(f"Before Filter: {count_before}")
    print(f"After Filter: {count_after}")



def generate_graph(pa_path):
    with open(pa_path, 'r') as f:
        pa_json = json.load(f)
    f.close()

    pa_graph = {}
    pa = pa_json["pa"]

    params = pa.keys()
    for p in params:
        pa_graph[p] = []

    for p in params:
        for p_name in pa:
            if p != p_name:
                p_with_backdash = "`" + p + "`"
                p_with_double_backdash = "``" + p + "``"
                if len(p) >= 3:
                    if p in pa[p_name] or p_with_backdash in pa[p_name] or p_with_double_backdash in pa[p_name]:
                        pa_graph[p].append(p_name)
                else:
                    if p_with_backdash in pa[p_name] or p_with_double_backdash in pa[p_name]:
                        pa_graph[p].append(p_name)
    return pa_graph


def has_cross_parameters(graph):
    for key in graph:
        if len(graph[key]) != 0:
            return True
    else:
        return False
    

def count_directories(path):
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return len(directories)

