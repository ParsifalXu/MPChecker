import os
import json
import libcst as cst


from tools.macros import INFO_DIR
from tools.macros import FLAG

import tools.modifier.FindInputParams
import tools.modifier.RemoveDetails
import tools.modifier.ReplaceUnprocessableAssign
import tools.modifier.ReplaceCallAndList
import tools.modifier.SimplifyTernaryOperator
import tools.modifier.XorToOrAnd
import tools.modifier.IsNotToEquals
import tools.modifier.UndefinedVariableCollector
import tools.modifier.AddParameters
import tools.modifier.AddReturn
import tools.modifier.FormatErrorReturn
import tools.modifier.LoopToIfTransformer
import tools.modifier.MultiAssignment

def translate_code(project):
    call_times = 0
    info_path = os.path.join(INFO_DIR, project)
    if not os.path.exists(info_path):
        print(f"{project}'s info files is not exist, please extract info first")
    folders = os.listdir(info_path)
    for folder in folders:
        if "DS_Store" in folder:
            continue
        try:
            call_times = translate_functions(project, folder, call_times)
        except SyntaxError as e:
            if "non-default argument follows default argument" in str(e):
                print(f"\n\nfunc:{folder}:\npath:{info_path}\n{e}\n\n")
            else:
                raise
    print(f"total call times: {call_times}")

def translate_functions(project, folder, call_times):
    if os.path.exists(f"{INFO_DIR}/{project}/{folder}/memberfunc"):
        memberfunc_folders = os.listdir(f"{INFO_DIR}/{project}/{folder}/memberfunc")
        for memberfunc in memberfunc_folders:
            if "DS_Store" in memberfunc:
                continue
            print(f"Translating Class: {folder}; Function: {memberfunc}")
            call_times+=1
            try:
                translate_single_function(project, folder, memberfunc)
            except Exception as e:
                print(f"Useless Constraint ---- {str(e)}")    
            trans_path = f"{INFO_DIR}/{project}/{folder}/memberfunc/{memberfunc}/{memberfunc}_trans.py"
            if not os.path.exists(trans_path):
                with open(trans_path, 'w') as fw:
                    pass
                fw.close()
    else:
        print(f"Translating Independent Function: {folder}")
        call_times+=1
        try:
            translate_single_function(project, folder, None)
        except Exception as e:
            print(f"Useless Constraint ---- {str(e)}")
        trans_path = f"{INFO_DIR}/{project}/{folder}/{folder}_trans.py"
        if not os.path.exists(trans_path):
            with open(trans_path, 'w') as fw:
                pass
            fw.close()
    return call_times
    

def translate_single_function(project, folder, memberfunc):
    if memberfunc:
        func_path = f"{INFO_DIR}/{project}/{folder}/memberfunc/{memberfunc}/{memberfunc}.py"
        pa_path = f"{INFO_DIR}/{project}/{folder}/memberfunc/{memberfunc}/{memberfunc}_pa.json"
        trans_path = f"{INFO_DIR}/{project}/{folder}/memberfunc/{memberfunc}/{memberfunc}_trans.py"
        parent_pa_path = f"{INFO_DIR}/{project}/{folder}/{folder}_pa.json"
        func = memberfunc
    else:
        func_path = f"{INFO_DIR}/{project}/{folder}/{folder}.py"
        pa_path = f"{INFO_DIR}/{project}/{folder}/{folder}_pa.json"
        trans_path = f"{INFO_DIR}/{project}/{folder}/{folder}_trans.py"
        parent_pa_path = f""
        func = folder

    print(f"func: {func_path}")
    with open(func_path, 'r') as f:
        source = f.read()
    f.close()

    source, input_params = tools.modifier.FindInputParams.extract_function_params(source)
    source = tools.modifier.RemoveDetails.remove_details(source)
    source = tools.modifier.SimplifyTernaryOperator.modify_ternary(source)
    source, init_params = tools.modifier.ReplaceUnprocessableAssign.replace_unprocessable_assign(source)
    try:
        source, call_name = tools.modifier.ReplaceCallAndList.replace_call(source)
    except:
        call_name = []
        print(f"ERRORRRRRRR HERE: {folder}")
    
    source, removed_assignment = tools.modifier.XorToOrAnd.modify_xor(source)
    source = tools.modifier.MultiAssignment.reconstuct_multiple_assignment(source)
    source = tools.modifier.IsNotToEquals.replace_comparison_operators(source)
    
    undefined_variables = tools.modifier.UndefinedVariableCollector.undef_var_collector(source)

    params = []
    if os.path.exists(pa_path):
        with open(pa_path, 'r') as fpa:
            pa_json = json.load(fpa)
        fpa.close()
        params = list(pa_json["pa"].keys())


    parent_params = []
    if os.path.exists(parent_pa_path):
        with open(parent_pa_path, 'r') as fppa:
            parent_pa_json = json.load(fppa)
        fppa.close()
        parent_params = list(parent_pa_json["pa"].keys())

    def merge_and_deduplicate(*lists):
        merged_set = set()
        for lst in lists:
            merged_set.update(lst)
        return list(merged_set)

    source, loops = tools.modifier.LoopToIfTransformer.replace_loop(source)
    merged_params = merge_and_deduplicate(call_name, removed_assignment, undefined_variables, init_params, parent_params, params, input_params, loops)
    try:
        source = tools.modifier.AddParameters.replace_function_parameters(source, func, merged_params)
        source = tools.modifier.AddReturn.add_return_to_end(source)
    except:
        print(f"ERRORRRRRRR HERE: {folder}")
    
    source = tools.modifier.FormatErrorReturn.format_error_ret(source)

    split_code = source.split('\n')
    last_return = split_code[-1]
    split_code[-1] = last_return.replace("'", "\"")
    source = '\n'.join(split_code)


    split_code = source.split('\n')
    new_split_code = []
    for i in range(0, len(split_code)-1):
        if split_code[i] != '':
            if "f'" in split_code[i]:
                new_line = split_code[i].replace("f'", 'f"').strip("'")
                new_line = new_line + '"'
                new_split_code.append(new_line)
            else:
                new_split_code.append(split_code[i])

    split_code = new_split_code
    last_line_1 = split_code[-1]
    last_line_2 = split_code[-2]

    if last_line_1.strip().startswith("return f"):
        leading_space_1 = len(last_line_1) - len(last_line_1.lstrip(" "))
        leading_space_2 = len(last_line_2) - len(last_line_2.lstrip(" "))
        if leading_space_1 == leading_space_2 and "return " in last_line_2:
            split_code[-2] = ''
    source = '\n'.join(split_code)

    with open(trans_path, 'w') as fw:
        fw.write(source.strip())
    fw.close()
