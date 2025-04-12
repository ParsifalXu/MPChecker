import os
import ast
import json

from tools.macros import INFO_DIR


def match_alignment(project):
    info_path = os.path.join(INFO_DIR, project)
    if not os.path.exists(info_path):
        print(f"Do not have project: {project}, please extract info first")
    folders = os.listdir(info_path)

    for folder in folders:
        if "DS_Store" in folder:
            continue
        check_match(project, folder)


def check_match(project, folder):
    if os.path.exists(f"{INFO_DIR}/{project}/{folder}/memberfunc"):
        memberfunc_folders = os.listdir(f"{INFO_DIR}/{project}/{folder}/memberfunc")
        for memberfunc in memberfunc_folders:
            if "DS_Store" in memberfunc:
                continue
            func_json_path = f"{INFO_DIR}/{project}/{folder}/memberfunc/{memberfunc}/{memberfunc}_pa.json"
            func_code_path = f"{INFO_DIR}/{project}/{folder}/memberfunc/{memberfunc}/{memberfunc}.py"
            single_match(func_json_path, func_code_path, folder)
    
    func_json_path = f"{INFO_DIR}/{project}/{folder}/{folder}_pa.json"
    func_code_path = f"{INFO_DIR}/{project}/{folder}/{folder}.py"
    single_match(func_json_path, func_code_path, folder)



def single_match(json_path, code_path, folder):
    if not os.path.exists(json_path):
        return

    with open(json_path, 'r') as fpa:
        pa_json = json.load(fpa)
    fpa.close()
    
    with open(code_path, 'r') as fc:
        source = fc.read()
    fc.close()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "Invalid code"
    if not tree.body:
        return "Empty code"

    first_node = tree.body[0]
    if isinstance(first_node, ast.ClassDef):
        params_in_code = extract_init_params(tree)
    elif isinstance(first_node, ast.FunctionDef):
        params_in_code = extract_function_params(tree)
    else:
        print("Other") 

    def remove_star_items(input_list):
        return [item for item in input_list if '*' not in item and 'kwargs' not in item and item != "cls"]

    def unique_items(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        unique_to_list1 = set1 - set2
        unique_to_list2 = set2 - set1
        return unique_to_list1.union(unique_to_list2)

    params_in_doc = list(pa_json["param"].keys())
    params_in_doc = remove_star_items(params_in_doc)
    params_in_code = remove_star_items(params_in_code)
    
    if set(params_in_doc) != set(params_in_code):
        print(f"--- MISMATCH : {folder} - {first_node.name} --- \nDoc: {params_in_doc}\nCode: {params_in_code}")
        print(f"Difference: {unique_items(params_in_doc, params_in_code)}\n")





def extract_init_params(tree):
    class_node = next((node for node in tree.body if isinstance(node, ast.ClassDef)), None)
    if class_node is None:
        return "No class found"

    init_node = next((node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"), None)
    if init_node is None:
        return "No __init__ method found"

    params = []
    for arg in init_node.args.args:
        if arg.arg not in ("self",):
            params.append(arg.arg)
    
    for arg in init_node.args.kwonlyargs:
        params.append(arg.arg)

    return params


def extract_function_params(tree):
    function_node = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
    if function_node is None:
        return "No function found"

    params = []
    for arg in function_node.args.args:
        if arg.arg not in ("self",):
            params.append(arg.arg)

    if function_node.args.vararg:
        pass

    for arg in function_node.args.kwonlyargs:
        params.append(arg.arg)

    if function_node.args.kwarg:
        pass

    return params