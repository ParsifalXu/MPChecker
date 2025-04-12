import os
import logging

from z3 import *
from tools.macros import FUZZWORDS
from tools.macros import UNSOLVED_WORDS
from colorlog import ColoredFormatter


def logger_maker():
    formatter = ColoredFormatter(
        '%(log_color)s%(name)s - %(levelname)s - %(message)s',
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'bold_blue',
            'INFO': 'cyan',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    return logger



def extract_constraint_expr(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        lines = f.readlines()
    lines.append("")
    lines.append("")
    constraints = []
    for i in range(len(lines)):
        if ("Logical format" in lines[i] or "Logical Format" in lines[i]) and ":" in lines[i]:
            if "```" in lines[i+1]:
                if "```" == lines[i+1].strip():
                    if not lines[i+2].strip() == "":
                        constraint = lines[i+2].split(":")[-1].strip("*`. \n")    
                else:
                    constraint = lines[i+1].split(":")[-1].strip("*`. \n")    
            else:
                constraint = lines[i].split(":")[-1].strip("*`. \n")
            if "->" in constraint:
                constraints.append(constraint)
    return constraints


def solve_expr(z3_expr):
    s = Solver()
    if z3_expr is not None:
        s.add(z3_expr)
        if s.check() == sat:
            return True
        else:
            return False

def solve_unsat_expr(z3_expr):
    s = Solver()
    if z3_expr is not None:
        s.add(z3_expr)
        if s.check() == unsat:
            return True
        else:
            return False

def is_contain_fuzzwords(text):
    words = FUZZWORDS["nonexistence"] + FUZZWORDS["existence"]
    text_lower = text.lower()
    return any("'" +word.lower() + "'" in text_lower for word in words)

def has_unsolvable_words(text):
    words = UNSOLVED_WORDS
    text_lower = text.lower()
    return any("'" +word.lower() + "'" in text_lower for word in words)


def have_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return not set1.isdisjoint(set2)

def all_related_error(list1, list2):
    return all("("+element+")" or element in list2 for element in list1)

def contains_symbols(params):
    symbols = [" ", ".", "(", ")", "[", "]", "{", "}"]
    for param in params:
        if any(symbol in param[0] or symbol in param[2] for symbol in symbols):
            return True
    return False

def find_related_error_params(path_params, error_params):
    related_params = []
    for ep in error_params:
        for pp in path_params:
            if ep == pp[0]:
                related_params.append(pp)
    return related_params