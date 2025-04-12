import os
import re
import time
import tools.parser

from tools.macros import INFO_DIR
from tools.auxiliary import *
from tools.simCalculator import *
from z3 import *
from pathlib import Path

RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

import tools.parser

logger = logger_maker()

def solve_constraints(project):
    project_path = os.path.join(INFO_DIR, project)
    
    for folder in os.listdir(project_path):
        try:
            print(f"\n===== Processing folder: {folder} =====")
            folder_path = os.path.join(project_path, folder)
            constraints = []
            constraint_file = os.path.join(folder_path, f"{folder}_constraints.txt")
            if os.path.exists(constraint_file):
                constraints = extract_constraint_expr(constraint_file)

            memberfunc_path = os.path.join(folder_path, "memberfunc")
            if os.path.exists(memberfunc_path):
                for memberfunc in os.listdir(memberfunc_path):
                    print(f"--- Solving memberfunc: {project}/{folder}/{memberfunc}")
                    path_file = os.path.join(memberfunc_path, memberfunc, f"{memberfunc}_path.txt")
                    check_constraints(constraints, path_file)
            else:
                print(f"--- Solving func: {project}/{folder}")
                path_file = os.path.join(folder_path, f"{folder}_path.txt")
                check_constraints(constraints, path_file)
                
        except Exception as e:
            print(f"Error processing {folder}: {str(e)}")
            continue
        print("All path files checked.")

def remove_invalid_parentheses(s):
    s = list(s)
    stack = []

    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                stack.pop()
            else:
                s[i] = '' 

    for i in stack:
        s[i] = ''

    return ''.join(s)


def check_constraints(constraints, path_file):
    """
    Check whether constraints are satisfied with the code-constraints
    """
    print(f"Checking constraints in {constraints}")
    if not constraints:
        logger.warning("Constraint file empty!")
        return
        
    try:
        print(f"Reading path file: {path_file}")
        with open(path_file, 'r') as f:
            paths = f.readlines()

        paths = [remove_invalid_parentheses(path.strip()) for path in paths]
        
    except IOError as e:
        logger.error(f"Failed to read path file {path_file}: {str(e)}")
        return
        
    invalid_constraints = []
    
    for idx, constraint in enumerate(constraints, 1):
        if has_unsolvable_words(constraint):
            logger.debug(f"Skipping constraint with unsolvable words: {constraint}")
            continue

        try:
            cons_pnames, cons_params, cons_expr, logic_expr = tools.parser.cons2z3(constraint)
            path_tetrads = []
            for p in paths:
                _, pathparams, _, _, _ = tools.parser.path2z3(p)
                path_tetrads.extend(pathparams)

            cons_sim = []
            for consp in cons_params:
                most_sim_tetrad, highest_sim = most_sim(consp, path_tetrads)
                cons_sim.append(highest_sim)
            sim_score = total_similarity(logic_expr, cons_sim)
            print(f"""\n~~~~~~~~~~\n{YELLOW}[Constraint #{idx}]{RESET}: {constraint}\n  - Parameter Names: {cons_pnames}\n  - Parameters: {cons_params}\n  - Expression: {cons_expr}\n  - Logic: {logic_expr}\n  - Similarity: {sim_score}\n~~~~~~~~~~""")
            if sim_score < 1 and sim_score > 0.85:
                logger.warning(f"Constraint similarity score is too high: {constraint}")

            if contains_symbols(cons_params):
                logger.debug(f"Skipping constraint with symbols: {constraint}")
                continue
            if is_contain_fuzzwords(constraint):
                fuzzy_solve(constraint, paths, path_file)
            else:
                combine_solve(cons_pnames, cons_expr, cons_params, constraint, paths, path_file)
        except Exception as e:
            invalid_constraints.append(f"{idx}. {constraint}")
            logger.error(f"Failed to process constraint {idx}: {str(e)}")
            continue
            
    if invalid_constraints:
        logger.warning(f"Invalid constraints found:\n" + "\n".join(invalid_constraints))
 

def check_array_in_file_list(array, path_file):
    if path_file.endswith("_path.txt"):
        new_filename = path_file[:-9] + ".py"
    else:
        return False
    with open(new_filename, 'r', encoding='utf-8') as f:
        file_contents_lines = [line.strip() for line in f] 
        file_contents = '\n'.join(file_contents_lines)
    res = all(item in file_contents for item in array)
    return all(item in file_contents for item in array)


def find_existence_vars(str):
    pattern = r"(\w+)\s*=\s*['\"](" + "|".join(map(re.escape, FUZZWORDS["existence"])) + r")['\"]"
    results = re.findall(pattern, str)
    vars_found = [x[0] for x in results]
    return vars_found

def find_nonexistence_vars(str):
    pattern = r"(\w+)\s*=\s*['\"](" + "|".join(map(re.escape, FUZZWORDS["nonexistence"])) + r")['\"]"
    results = re.findall(pattern, str)
    vars_found = [x[0] for x in results]
    return vars_found

def is_existence(text):
    words = FUZZWORDS["existence"]
    text_lower = text.lower()
    return any("'" +word.lower() + "'" in text_lower for word in words)

def is_nonexistence(text):
    words = FUZZWORDS["nonexistence"]
    text_lower = text.lower()
    return any("'" +word.lower() + "'" in text_lower for word in words)


def fuzzy_solve(constraint, paths, path_file):
    print(f"constraint: {constraint}")
    print(f"paths: {paths}")
    print(f"path_file: {path_file}")
    first_part = constraint.split("->")[0]
    second_part = constraint.split("->")[1]
    print(f"first_part: {first_part}")
    print(f"second_part: {second_part}")


    exist_flag = True
    if is_existence(second_part):
        existence_vars = find_existence_vars(second_part)
    elif is_nonexistence(second_part):
        exist_flag = False
        existence_vars = find_nonexistence_vars(second_part)

    cons_pnames, cons_params, cons_expr, logic_expr = tools.parser.cons2z3(first_part)
    print(f"cons_pnames: {cons_pnames}")
    print(f"cons_params: {cons_params}")
    print(f"cons_expr: {cons_expr}")
    print(f"logic_expr: {logic_expr}")
    print(f"exist_flag: {exist_flag}")
    print(f"existence_vars: {existence_vars}")

    if not check_array_in_file_list(cons_pnames, path_file):
        return

    results = []
    existence_expr = BoolVal(True)
    for idx, path in enumerate(paths, 1):

        pos = path.rfind('->')
        if pos != -1:
            leftpath = path[:pos].strip()
        if exist_flag:
            if all(var + " " in leftpath for var in existence_vars):
                existence_expr = BoolVal(True)
            else:
                existence_expr = BoolVal(False)
        else:
            if all(var + " " not in leftpath for var in existence_vars):
                existence_expr = BoolVal(True)
            else:
                existence_expr = BoolVal(False)

        print(f"existence_expr: {existence_expr}")

        try:
            path_pnames, path_params, path_expr, error_params, path_logic = tools.parser.path2z3(path)
        except Exception as e:
            results.append(True)
            continue
        print(f"""\n    {YELLOW}|==>{RESET} [Path #{idx}]: {path}     - Path Parameter Names: {path_pnames}\n     - Path Parameters: {path_params}\n     - Path Expression: {path_expr}\n     - Error_Params: {error_params}\n     - Logic: {path_logic}\n""")
        
        results = strategy1(results, path_pnames, cons_pnames, error_params, path_params, cons_expr, path_expr, existence_expr)


    print(f"normal results: {results}")
    if results and not all(results):
        print("\n")
        print(f"{RED}[ BAD CONSTRAINT ]{RESET}")
        print("#"*50)
        parts = Path(path_file).parts
        idx = parts.index("info")
        print(f"Library: {parts[idx+1]}")
        if parts[idx+3] and parts[idx+3] == "memberfunc":
            print(f"Class: {parts[idx+2]}")
            print(f"Memberfunc: {parts[idx+4]}")
        else:
            print(f"Function: {parts[idx+2]}")
        print(f"Constraint: {constraint}")
        print(f"path file: {path_file}")
        print("#"*50)
        print("\n")
        exit(0)


def combine_solve(cons_pnames, cons_expr, cons_params, constraint, paths, path_file):
    if len(cons_params) <= 1:
        return
    if not check_array_in_file_list(cons_pnames, path_file):
        return

    results = []
    existence_expr = BoolVal(True)
    for idx, path in enumerate(paths, 1):
        try:
            path_pnames, path_params, path_expr, error_params, path_logic = tools.parser.path2z3(path)
        except Exception as e:
            results.append(True)
            continue
        results = strategy1(results, path_pnames, cons_pnames, error_params, path_params, cons_expr, path_expr, existence_expr)

    print(f"normal results: {results}")
    if results and not any(results):
        print("\n")
        print(f"{RED}[ BAD CONSTRAINT ]{RESET}")
        print("#"*50)
        parts = Path(path_file).parts
        idx = parts.index("info")
        print(f"Library: {parts[idx+1]}")
        if parts[idx+3] and parts[idx+3] == "memberfunc":
            print(f"Class: {parts[idx+2]}")
            print(f"Memberfunc: {parts[idx+4]}")
        else:
            print(f"Function: {parts[idx+2]}")
        print(f"Constraint: {constraint}")
        print(f"path file: {path_file}")
        print("#"*50)
        print("\n")
        exit(0)





def strategy1(results, path_pnames, cons_pnames, error_params, path_params, cons_expr, path_expr, existence_expr):
    if all(cp in path_pnames for cp in cons_pnames):
        if error_params:
            error_related_params = find_related_error_params(path_params, error_params)
            err_path_expr = None
            for erp_tetrad in error_related_params:
                err_expr = tools.parser.tetrad2z3expr(erp_tetrad)
                if err_path_expr == None:
                    err_path_expr = err_expr
                else:
                    err_path_expr = And(err_path_expr, err_expr)
            results.append(not solve_expr(And(And(cons_expr, err_path_expr), existence_expr)))
        else:
            results.append(solve_expr(And(And(cons_expr, path_expr), existence_expr)))
    return results

def strategy2(results, path_pnames, cons_pnames, error_params, path_params, cons_expr, path_expr, existence_expr):
    if all(cp in path_pnames for cp in cons_pnames):
        if error_params:
            error_related_params = find_related_error_params(path_params, error_params)
            err_path_expr = None
            for erp_tetrad in error_related_params:
                err_expr = tools.parser.tetrad2z3expr(erp_tetrad)
                if err_path_expr == None:
                    err_path_expr = err_expr
                else:
                    err_path_expr = And(err_path_expr, err_expr)
            results.append(not solve_unsat_expr(And(And(cons_expr, Not(path_expr)), existence_expr)))
        else:
            results.append(solve_unsat_expr(And(And(cons_expr, Not(path_expr)), existence_expr)))
    else:
        results.append(True)
    return results
