import ast
import astor

class FormatErrorReturn(ast.NodeTransformer):
    def __init__(self):
        self.stack = []

    def visit_If(self, node):
        condition = self.get_condition(node.test)
        self.stack.append(condition)
        if isinstance(node.body[0], ast.Return) and isinstance(node.body[0].value, ast.Str):
            return_value = node.body[0].value.s
            if return_value == "ERROR_END" or return_value == "ASSERT_END" or return_value == "WARNING_END":
                condition_str = ')_('.join(self.stack)
                new_return_value = f"({condition_str})_{return_value}"
                node.body[0].value = ast.Constant(value=new_return_value)
                if self.stack:
                    self.stack.pop(-1)
        self.generic_visit(node)
        if node.orelse:
            for orelse_node in node.orelse:
                if isinstance(orelse_node, ast.If):
                    self.visit(orelse_node) 
                elif isinstance(orelse_node, ast.Return) and isinstance(orelse_node.value, ast.Str):
                    return_value = orelse_node.value.s
                    if return_value == "ERROR_END" or return_value == "ASSERT_END" or return_value == "WARNING_END":
                        cond = self.get_condition(node.test)
                        if self.stack:
                            if self.stack[-1] != cond:
                                self.stack.append('not'+cond)
                        condition_str = ')_('.join(self.stack)
                        new_return_value = f"({condition_str})_{return_value}"
                        orelse_node.value = ast.Constant(value=new_return_value)
                        if self.stack:
                            self.stack.pop(-1) 
                if self.stack:
                    self.stack.pop(-1)    
        return node

    def get_condition(self, test):
        if isinstance(test, ast.BoolOp):
            return ')_('.join(self.get_condition(op) for op in test.values)
        elif isinstance(test, ast.Compare):
            return ''.join(self.get_condition(operand) for operand in [test.left] + test.comparators)
        elif isinstance(test, ast.Name):
            return test.id
        return ''


def format_error_ret(code):
    tree = ast.parse(code)
    transformer = FormatErrorReturn()
    new_tree = transformer.visit(tree)
    return ast.unparse(new_tree)