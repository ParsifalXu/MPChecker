import os
import re
import json

from tools.macros import INFO_DIR
from tools.macros import FLAG


def find_params_and_attributes(project):
    info_path = os.path.join(INFO_DIR, project)
    if not os.path.exists(info_path):
        print(f"{project}'s info files is not exist, please extract info first")
    folders = os.listdir(info_path)
    for folder in folders:
        if "DS_Store" in folder:
            continue
        find_pa_content(project, folder)



def find_pa_content(project, folder):
    print(f"Processing ----- {folder}")
    if os.path.exists(f"{INFO_DIR}/{project}/{folder}/memberfunc"):
        memberfunc_folders = os.listdir(f"{INFO_DIR}/{project}/{folder}/memberfunc")
        for memberfunc in memberfunc_folders:
            if "DS_Store" in memberfunc:
                continue
            print(f"Processing memberfunc: {memberfunc}")
            file_path = f"{INFO_DIR}/{project}/{folder}/memberfunc/{memberfunc}/{memberfunc}_docstring.txt"
            output_path = f"{INFO_DIR}/{project}/{folder}/memberfunc/{memberfunc}/{memberfunc}_pa.json"
            parse_and_save(project, file_path, output_path)
    
    file_path = f"{INFO_DIR}/{project}/{folder}/{folder}_docstring.txt"
    output_path = f"{INFO_DIR}/{project}/{folder}/{folder}_pa.json"
    parse_and_save(project, file_path, output_path)

    
def parse_and_save(project, file_path, output_path):
    if not os.path.exists(file_path):
        return
    try:
        if FLAG[project] == "Args:": 
            args, attributes = parse_google_style_docstring(file_path)
        elif FLAG[project] == "Parameters" or FLAG[project] == "Parameters:": 
            args, attributes = parse_numpy_style_docstring(file_path)
    except:
        print("Format Wrong")
        return
    try:
        if not args:
            return 
    except:
        return 
    save_to_json(project, args, attributes, output_path)



def save_to_json(project, args, attributes, output_path):
    def remove_keys_with_stars(input_dict):
        keys_to_remove = [key for key in input_dict.keys() if '*' in key or (' ' in key and ',' not in key) or '.' in key or '[' in key or ']' in key]
        for key in keys_to_remove:
            del input_dict[key]
        return input_dict
    
    def clean_dict_keys(input_dict):
        cleaned_dict = {}
        for key, value in input_dict.items():
            paren_index = key.find('(')
            if paren_index > 0:
                new_key = key[:paren_index].strip()
            else:
                new_key = key
            cleaned_dict[new_key] = value
        return cleaned_dict

    args = remove_keys_with_stars(clean_dict_keys(args))
    attributes = remove_keys_with_stars(clean_dict_keys(attributes))
    
    data = {
        "param": args,
        "attr": attributes,
        "pa": {**args, **attributes}
    }

    def split_keys_with_commas(data):
        pattern1 = r'\{([^\{\}]+)\}_([a-zA-Z0-9_]+)' 
        pattern2 = r'\(([^()]+)\)' 
        if isinstance(data, dict):
            new_data = {}
            for key, value in data.items():
                if ',' in key:
                    print(f"key: {key}")
                    matches1 = re.findall(pattern1, key)
                    matches2 = re.findall(pattern2, key)
                    print(f"matches2: {matches2}")
                    if matches1:
                        items, suffix = matches1[0]
                        items = items.split(',')
                        items = [item.strip() for item in items]
                        expanded_items = [f"{item}_{suffix}" for item in items]
                        for ei in expanded_items:
                            new_data[ei.strip()] = value
                    elif matches2:
                        if project == "scipy" or project == "example":
                            items = matches2[0].split(',')
                            items = [item.strip() for item in items]
                            combine_key = '_and_'.join(items)
                            new_data[combine_key.strip()] = value
                        else:
                            items = matches2[0].split(',')
                            items = [item.strip() for item in items]
                            for i in items:
                                new_data[i.strip()] = value
                    else:
                        keys = key.split(',')
                        for k in keys:
                            new_data[k.strip()] = value
                else:
                    new_data[key] = split_keys_with_commas(value)
            return new_data
        elif isinstance(data, list):
            return [split_keys_with_commas(item) for item in data]
        else:
            return data
    
    new_data = split_keys_with_commas(data)
        

    with open(output_path, 'w') as json_file:
        json.dump(new_data, json_file, indent=4)


def parse_google_style_docstring(file_path):
    args = {}
    attributes = {}
    section = None
    current_key = None
    current_value = []

    first_arg = False
    leading_space = 0

    def save_current_param():
        nonlocal current_key, current_value
        if current_key is not None:
            combined_value = " ".join(current_value).strip()
            combined_value = re.sub(r'\s*,\s*', ', ', combined_value)  
            if section == "args":
                args[current_key] = combined_value
            elif section == "attributes":
                attributes[current_key] = combined_value
            current_key = None
            current_value = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.rstrip()
            if line.startswith("Args:"):
                save_current_param()
                section = "args"
            elif line.startswith("Attributes:"):
                save_current_param()
                section = "attributes"
            elif line.startswith("Returns:"):
                save_current_param()
                section = None
            elif line.startswith("returns"):
                save_current_param()
                section = None
            elif line.startswith("Call Args:"):
                save_current_param()
                section = None
            elif line.startswith("Output:"):
                save_current_param()
                section = None
            elif line.startswith("Examples:"):
                save_current_param()
                section = None
            elif line.startswith("Example:"):
                save_current_param()
                section = None
            elif line.startswith("Raises:"):
                next(file)
                save_current_param()
                section = None
            elif line.startswith("Note:"):
                next(file)
                save_current_param()
                section = None
            elif line.startswith("References:"):
                save_current_param()
                section = None
            elif line.startswith("Call arguments:"):
                save_current_param()
                section = None
            elif line.startswith("Input shape:"):
                save_current_param()
                section = None
            elif line.startswith("Output shape:"):
                save_current_param()
                section = None
            elif section and ":" in line:
                leading_space = len(line) - len(line.lstrip(" "))
                if not first_arg:
                    first_arg_leading_space = leading_space
                    first_arg = True
                
                if leading_space == first_arg_leading_space: 
                    save_current_param()
                    current_key, value = line.split(":", 1)
                    current_key = current_key.strip()
                    current_value.append(value.strip())
                else:
                    current_value.append(line.strip())
            elif section and current_key is not None:
                current_value.append(line.strip())

    save_current_param()  
    return args, attributes


def parse_numpy_style_docstring(file_path):
    args = {}
    attributes = {}
    section = None
    current_key = None
    current_value = []

    first_arg = False
    leading_space = 0

    def save_current_param():
        nonlocal current_key, current_value
        if current_key is not None:
            combined_value = " ".join(current_value).strip()
            combined_value = re.sub(r'\s*,\s*', ', ', combined_value)  
            if section == "args":
                args[current_key] = combined_value
            elif section == "attributes":
                attributes[current_key] = combined_value
            current_key = None
            current_value = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.rstrip()
            if line.startswith("Parameters"):
                next(file)
                save_current_param()
                section = "args"
            elif line.startswith("Attributes"):
                next(file)
                save_current_param()
                section = "attributes"
            elif line.startswith("Returns"):
                next(file)
                save_current_param()
                section = None
            elif line.startswith("return") and ":" not in line:
                next(file)
                save_current_param()
                section = None
            elif line.startswith("Examples"):
                next(file)
                save_current_param()
                section = None
            elif line.startswith("Example"):
                next(file)
                save_current_param()
                section = None
            elif line.startswith("Yields"):
                next(file)
                save_current_param()
                section = None
            elif line.startswith("Notes"):
                next(file)
                save_current_param()
                section = None
            elif line.startswith("See Also"):
                next(file)
                save_current_param()
                section = None
            elif line.startswith("Raises"):
                next(file)
                save_current_param()
                section = None
            elif line.startswith("References"):
                next(file)
                save_current_param()
                section = None
            elif section and ":" in line:
                leading_space = len(line) - len(line.lstrip(" "))
                if not first_arg:
                    first_arg_leading_space = leading_space
                    first_arg = True
                
                if leading_space == first_arg_leading_space: 
                    save_current_param()
                    current_key, value = line.split(":", 1)
                    current_key = current_key.strip()
                    current_value.append(value.strip())
                else:
                    current_value.append(line.strip())
            elif section and current_key is not None:
                current_value.append(line.strip())

    save_current_param()  
    return args, attributes


