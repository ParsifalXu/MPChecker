import re
from z3 import *

def cons2z3(constraint):
    constraint = constraint.strip()

    def ensure_no_quotes_around_literals(input_str):
        def remove_quotes(match):
            return match.group(1)
        pattern = r'["\'](None|False|True)["\']'
        return re.sub(pattern, remove_quotes, input_str)

    constraint = ensure_no_quotes_around_literals(constraint).replace("None", "'None'").replace("False", "'False'").replace("True", "'True'")
    pattern = re.compile(r"(\w+)\s*(=|!=|<=|>=|<|>)\s*(?:'(\w+)'|(\w+))")
    matches = pattern.findall(constraint)

    if len(matches) == 1:
        return [matches[0][0]], matches, tetrad2z3expr(matches[0]), "{}"

    def remove_closest_right_parenthesis(text):
        text = list(text)
        idx = 0
        while idx < len(text):
            if text[idx] == '[':
                for j in range(idx + 1, len(text)):
                    if text[j] == ')':
                        text[j] = "]"
                        break
            idx += 1
        return ''.join(text)
    
    lexpr = remove_closest_right_parenthesis(pattern.sub("{}", constraint).replace(" ","").replace("!(", "["))
    logic_expr = lexpr.replace("({})", "{}").replace("->", "^").replace("||", "|").replace("&&", "^")

    def add_index(s):
        idx = 0
        def replacer(match):
            nonlocal idx
            result = f'{{{idx}}}' 
            idx += 1
            return result

        return re.sub(r'\{\}', replacer, s)

    logic_expr = add_index(logic_expr)
    logic_expr = logic_expr.replace("^(call_len-1)!=n_bins_list_13^", "^").replace("^(call_len-1)=n_bins_list_13^", "^").replace("^(subsample=-1)^", "^")
    
    params = []
    opposite_table = {
        ">": "<",
        "<": ">",
        ">=": "<=",
        "<=": ">="
    }
    for match in matches:
        if match[0].isnumeric():
            params.append((match[3], opposite_table[match[1]], '',match[0]))
        elif match[3] != '' and not match[3].isnumeric():
            params.append((match[0], match[1], match[3], ''))
        else:
            params.append(match)

    res_expr = None
    res_expr = trans2expr(logic_expr, params)
    pnames = [p[0] for p in params]

    return pnames, params, res_expr, logic_expr

def calc(a, b, op):
    if op == "^":
        return And(a, b)
    elif op == "|":  
        return Or(a, b)


def trans2expr(expr, params):

    def find_matching_paren(s, left_index):
        if s[left_index] != '(':
            return -1

        stack = []
        n = len(s)

        for i in range(left_index, n):
            if s[i] == '(':
                stack.append(i)
            elif s[i] == ')':
                stack.pop()
                if not stack:
                    return i
        return -1

    exs = []
    ops = []
    i = 0
    n = len(expr)

    while i < n:
        if expr[i] == "(":
            k = find_matching_paren(expr, i)
            returned = trans2expr(expr[i+1:k], params)
            exs.append(returned)
            i = k + 1
        elif expr[i] == "[":
            ops.append(expr[i])
            i += 1
        elif expr[i] == "^":
            ops.append(expr[i])
            i += 1
        elif expr[i] == "|":
            ops.append(expr[i])
            i += 1
        elif expr[i] == "{":
            for j in range(i, n):
                if expr[j] == "}":
                    idx = int(expr[i:j+1].strip("{}"))
                    ex = tetrad2z3expr(params[idx])
                    exs.append(ex)
                    i = j + 1
                    break
        elif expr[i] == "]":
            if ops[-1] == "[":
                ex = exs.pop()
                exs.append(Not(ex))                 
            ops.pop()
            i += 1
        else:
            i += 1

    while ops:
        op = ops.pop()
        expr_b = exs.pop()
        expr_a = exs.pop()
        exs.append(calc(expr_a, expr_b, op))

    return exs[0] if exs else None


def path2z3(path):
    path = path.replace(" ", "")
    condition_path = path[:path.rfind("->")]
    last_str = path[path.rfind("->")+2:]
    error_params = []
    if "_END" in last_str:
        parts = last_str.strip("' ").split(")_")
        for part in parts:
            if "(" in part:
                error_params.append(part.strip("()"))
    else:        
        condition_path = condition_path + "->" + last_str.strip('" \n')

    path_pnames, path_params, path_expr, logic = cons2z3(condition_path)
    return path_pnames, path_params, path_expr, error_params, logic





def find_contents_in_parentheses(s):
    pattern = r'\((.*?)\)'
    matches = re.findall(pattern, s)
    return matches

def tetrad2z3expr(tetrad):
    op = tetrad[1]
    if op == "=":
        return tetrad2Expr(tetrad).equal()
    elif op == "!=":
        return tetrad2Expr(tetrad).notequal()
    elif op == "<":
        return tetrad2Expr(tetrad).less()
    elif op == "<=":
        return tetrad2Expr(tetrad).lessequal()
    elif op == ">":
        return tetrad2Expr(tetrad).greater()
    elif op == ">=":
        return tetrad2Expr(tetrad).greaterequal()


class tetrad2Expr():
    def __init__(self, tetrad):
        self.var = tetrad[0]
        self.op = tetrad[1]
        self.str_val = tetrad[2]
        self.nonstr_val = tetrad[3]

    def equal(self):
        if self.str_val:
            if self.str_val == 'True':
                return Real(self.var) != 0
            elif self.str_val == 'False':
                return Real(self.var) == 0
            else:
                return String(self.var) == StringVal(self.str_val)
        elif self.nonstr_val:
            return Real(self.var) == RealVal(self.nonstr_val)
    
    def notequal(self):
        if self.str_val:
            if self.str_val == 'True':
                return Real(self.var) == 0
            elif self.str_val == 'False':
                return Real(self.var) != 0
            else:
                return String(self.var) != StringVal(self.str_val)
        elif self.nonstr_val:
            return Real(self.var) != RealVal(self.nonstr_val)

    def less(self):
        if self.nonstr_val:
            return Real(self.var) < RealVal(self.nonstr_val)
        elif self.str_val:
            return Real(self.var) < Real(self.str_val)

    def lessequal(self):
        if self.nonstr_val:
            return Real(self.var) <= RealVal(self.nonstr_val)
        elif self.str_val:
            return Real(self.var) <= Real(self.str_val)

    def greater(self):
        if self.nonstr_val:
            return Real(self.var) > RealVal(self.nonstr_val)
        elif self.str_val:
            return Real(self.var) > Real(self.str_val)

    def greaterequal(self):
        if self.nonstr_val:
            return Real(self.var) >= RealVal(self.nonstr_val)
        elif self.str_val:
            return Real(self.var) >= Real(self.str_val)
