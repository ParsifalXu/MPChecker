import libcst as cst
import libcst.matchers as m
from libcst import parse_module, Module


class ParameterReplacer(cst.CSTTransformer):
    def __init__(self, function_name, new_params):
        self.function_name = function_name
        self.new_params = new_params

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        if original_node.name.value == self.function_name:
            new_params = cst.Parameters(
                params=[
                    cst.Param(name=cst.Name(param_name))
                    for param_name in self.new_params
                ]
            )
            return updated_node.with_changes(params=new_params)
        return updated_node
    
    def leave_SimpleString(self, original_node, updated_node):
        if updated_node.value.startswith('"""'):
            new_value = "'''" + updated_node.value[1:-1].replace("'", "\\'") + "'''"
            return updated_node.with_changes(value=new_value)
        if updated_node.value.startswith('"'):
            new_value = "'" + updated_node.value[1:-1].replace("'", "\\'") + "'"
            return updated_node.with_changes(value=new_value)
        return updated_node


def replace_function_parameters(code: str, function_name: str, new_params: list) -> str:
    tree = parse_module(code)
    transformer = ParameterReplacer(function_name, new_params)
    updated_tree = tree.visit(transformer)
    return updated_tree.code
