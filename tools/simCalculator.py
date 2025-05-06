import numpy as np
operator_table = {
    "<": [1, 0, 0, 1, 0],
    ">": [1, 0, 1, 0, 0],
    "<=": [1, 1, 0, 1, 0],
    ">=": [1, 1, 1, 0, 0],
    "=": [0, 1, 0, 0, 0],
    "!=": [0, 1, 0, 0, 1]
}

def levenshtein_distance(s1, s2):
    dp = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])  
    return dp[len(s1)][len(s2)]

def normalized_levenshtein_distance(str1, str2):
    lev_distance = levenshtein_distance(str1, str2)
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 0.0
    normalized_distance = 1 - lev_distance / max_len
    return normalized_distance


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)



def similarity(p1, p2):
    alpha = 0.25
    beta = 0.5
    ld1 = normalized_levenshtein_distance(p1[0], p2[0])
    opcs = cosine_similarity(operator_table[p1[1]], operator_table[p2[1]])
    ld2 = normalized_levenshtein_distance(p1[2], p2[2])
    return round(alpha * ld1 + beta * opcs + alpha * ld2, 2)
    # return round((ld1 + opcs + ld2) / 3, 2)


def most_sim(tetrad, list):
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    if is_number(tetrad[3]):
        return tetrad, 1.0
    if tetrad[0] == "batch_size":
        return tetrad, 1.0
    if tetrad[2].startswith("numpy.take("):
        return tetrad, 1.0
    # if tetrad[2] == "True" or tetrad[2] == "False":
    #     return tetrad, 1.0
    
    highest_sim = 0
    most_sim_tetrad = None
    triple = tuple(it for it in tetrad if it)
    triple_list = [(left, op, string_value or non_string_value) for left, op, string_value, non_string_value in list]
    for i in range(len(triple_list)):    
        cur_sim = similarity(triple, triple_list[i])    
        if cur_sim > highest_sim:
            highest_sim = cur_sim
            most_sim_tetrad = list[i]
    return most_sim_tetrad, highest_sim


def calc_similarity(a, b, op):
    if op == "^":
        return min(a, b)
    elif op == "|":  
        return max(a, b)

def total_similarity(expr, sims):

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
            returned = total_similarity(expr[i+1:k], sims)
            print(returned)
            exs.append(returned)
            i = k + 1
        elif expr[i] == "[" and expr[i+1] == "{":
            for j in range(i+2, n):
                if expr[j] == "}" and expr[j+1] == "]":
                    idx = int(expr[i:j+1].strip("{}[]"))
                    exs.append(sims[idx]*0.85)
                    i = j + 2
                    break
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
                    exs.append(sims[idx])
                    i = j + 1
                    break
        elif expr[i] == "]":
            if ops[-1] == "[":
                ex = exs.pop()
                exs.append(1-ex)                 
            ops.pop()
            i += 1 

    while ops:
        op = ops.pop()
        expr_b = exs.pop()
        expr_a = exs.pop()
        exs.append(calc_similarity(expr_a, expr_b, op))

    return exs[0] if exs else None