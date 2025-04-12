import re
import logging
from graphviz import Source
import pygraphviz as pgv

from pyexsmt import pred_to_smt, get_concr_value, match_smt_type
from pyexsmt.symbolic_types import SymbolicObject
from pyexsmt.symbolic_types.symbolic_object import to_pysmt, is_instance_userdefined_and_newclass

from pysmt.shortcuts import *

class Result(object):
    def __init__(self, path):
        self.path = path
        self.generated_inputs = []
        self.execution_return_values = []
        self.list_rep = None
        self.curr_id = 0

    def record_inputs(self, inputs):
        inputs = [(k, get_concr_value(inputs[k])) for k in inputs]
        self.generated_inputs.append(inputs)
        logging.debug("RECORDING INPUTS: %s", inputs)

    def record_output(self, ret):
        logging.info("RECORDING EFFECT: %s -> %s", self.path.current_constraint, ret)
        self.path.current_constraint.effect = ret
        if isinstance(ret, SymbolicObject):
            ret = ret.get_concr_value()
        self.execution_return_values.append(ret)

    def to_dot(self, filename, mapping):
        header = "digraph {\n"
        footer = "}\n"
        if self.list_rep is None:
            self.list_rep = self._to_list_rep(self.path.root_constraint)
        dot = self._to_dot(self.list_rep)
        dot = header + dot + footer
        for key, value in mapping.items():
            dot = dot.replace(str(value), f"'{key}'")
        s = Source(dot, filename=filename+".dot", format="png")
        s.view()

    def _to_dot(self, list_rep):
        curr = self.curr_id
        if isinstance(list_rep, list) and len(list_rep) == 3:
            rep = list_rep[0]
            dot = "\"%s%d\" [ label=\"%s\" ];\n" % (rep, curr, rep)
            self.curr_id += 1

            for slot in range(1, 3):
                child = list_rep[slot]
                if child is None:
                    continue
                crep = child[0] if isinstance(child, list) else to_pysmt(child)
                crep = str(crep).replace('"', '\\\"')
                dot += "\"%s%d\" -> \"%s%d\" [ label=\"%d\" ];\n" \
                        %(rep, curr, crep, self.curr_id, slot%2)
                dot += self._to_dot(child)
            return dot
        elif list_rep is not None:
            list_rep = to_pysmt(list_rep)
            list_rep = str(list_rep).replace('"', '\\\"')
            temp = "\"%s%d\" [ label=\"%s\" ];\n" % (list_rep, curr, list_rep)
            self.curr_id += 1
            return temp
        else:
            return ""

    def to_path(self, filename, mapping):
        header = "digraph {\n"
        footer = "}\n"
        if self.list_rep is None:
            self.list_rep = self._to_list_rep(self.path.root_constraint)        
        dot = self._to_path(self.list_rep)
        dot = header + dot + footer
        for key, value in mapping.items():
            dot = dot.replace(str(value), f"'{key}'")
        graph = pgv.AGraph(dot)
        node_indices = {node: i for i, node in enumerate(graph.nodes())}
        start_node = list(graph.nodes())[0]
        paths = []
        self.extract_paths(graph, start_node, [(start_node, None)], paths)
        for path in paths:                
            formatted_path = " -> ".join(f"{node.get_name()[:-(len(str(node_indices[node])))]} ({'True' if condition else 'False'})" for node, condition in path)
            parts = formatted_path.split("->")
            result = []
            for part in parts:
                match = re.match(r"\((.*?)\) \((True|False)\)", part.strip())
                if match:
                    result.append((match.group(1), match.group(2)))
                else:
                    match = re.match(r"(.*?) \((True|False)\)", part.strip())
                    if match:
                        result.append((match.group(1).strip(), match.group(2)))

            def adjust(condition, flag):
                if condition.isnumeric():
                    return condition
                if not ((flag == 'True') ^ (condition.startswith("!"))):
                    if '!=' in condition:
                        adjusted_condition = condition.strip(" ()!").replace('!=', '=')
                    elif '==' in condition:
                        adjusted_condition = condition.strip(" ()!").replace('==', '!=')
                    elif '<=' in condition:
                        adjusted_condition = condition.strip(" ()!").replace('<=', '>')
                    elif '>=' in condition:
                        adjusted_condition = condition.strip(" ()!").replace('>=', '<')
                    elif '<' in condition:
                        adjusted_condition = condition.strip(" ()!").replace('<', '>=')
                    elif '>' in condition:
                        adjusted_condition = condition.strip(" ()!").replace('>', '<=')
                    else:
                        adjusted_condition = condition.strip(" ()!").replace('=', '!=')
                else:
                    adjusted_condition = condition.strip(" ()!")
                
                return f"({adjusted_condition})"

            modified_list = []
            for i in range(0, len(result)-1):
                condition = result[i][0]
                flag = result[i+1][1]
                modified_list.append(adjust(condition, flag))
        
            
            if "=" in result[-1][0] and "^" not in result[-1][0]:
                condition = result[-1][0]
                flag = 'False' if condition.count("!") % 2 == 0 else 'True'
                modified_list.append(adjust(condition, flag))
            else:
                modified_list.append(result[-1][0])

            print(" -> ".join(modified_list))


    def extract_paths(self, graph, node, path, paths):
        successors = graph.successors(node)
        if not successors:
            paths.append(path)
            return
        for successor in successors:
            edge = graph.get_edge(node, successor)
            if edge:
                condition = edge.attr['label'] == '1'
                new_path = path + [(successor, condition)]
                self.extract_paths(graph, successor, new_path, paths)

    def _to_path(self, list_rep):
        curr = self.curr_id
        if isinstance(list_rep, list) and len(list_rep) == 3:
            rep = list_rep[0]
            dot = "\"%s%d\" [ label=\"%s\" ];\n" % (rep, curr, rep)
            self.curr_id += 1

            for slot in range(1, 3):
                child = list_rep[slot]
                if child is None:
                    continue
                crep = child[0] if isinstance(child, list) else to_pysmt(child)
                crep = str(crep).replace('"', '\\\"')
                dot += "\"%s%d\" -> \"%s%d\" [ label=\"%d\" ];\n" \
                        %(rep, curr, crep, self.curr_id, slot%2)
                dot += self._to_path(child)
            return dot
        elif list_rep is not None:
            list_rep = to_pysmt(list_rep)
            list_rep = str(list_rep).replace('"', '\\\"')
            temp = "\"%s%d\" [ label=\"%s\" ];\n" % (list_rep, curr, list_rep)
            self.curr_id += 1
            return temp
        else:
            return ""







    def to_summary(self, unknown=Symbol('Unknown', INT)):
        if self.list_rep is None:
            self.list_rep = self._to_list_rep(self.path.root_constraint)
        summary = self._to_summary(self.list_rep, unknown)
        return summary

    def _to_summary(self, list_rep, unknown):
        if isinstance(list_rep, list) and len(list_rep) == 3:
            return Ite(list_rep[0], self._to_summary(list_rep[1], unknown),\
                        self._to_summary(list_rep[2], unknown))
        elif list_rep is not None:
            if isinstance(list_rep, SymbolicObject) or not is_instance_userdefined_and_newclass(list_rep):
                return match_smt_type(to_pysmt(list_rep), unknown.get_type())
            else:
                print(list_rep, type(list_rep))
                raise TypeError("Summaries don't support object returns")
        else:
            return unknown

    def _to_list_rep(self, node):
        if node is None:
            return None
        children = node.children
        if len(children) == 2:
            if node.children[0].predicate.symtype.symbolic_eq(node.children[1].predicate.symtype):
                left = node.children[0] if node.children[0].predicate.result else node.children[1]
                right = node.children[1] if not node.children[1].predicate.result else node.children[0]
                return [pred_to_smt(left.predicate), self._to_list_rep(left), self._to_list_rep(right)]
            else:
                raise ValueError("Two children of a constraint should have the same predicate!")
        elif len(node.children) == 1:
            return [pred_to_smt(node.children[0].predicate), self._to_list_rep(node.children[0]), None]
        elif len(children) == 0:
            return node.effect

        raise ValueError("Should not be possible! Can't have more than two children.")
