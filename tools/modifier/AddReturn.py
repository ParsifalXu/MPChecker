import ast

class AddReturn(ast.NodeTransformer):
    def __init__(self):
        self.func_args = {}
        self.cur_func = ""

    def visit_FunctionDef(self, node):
        self.cur_func = node.name
        args = [arg.arg for arg in node.args.args]
        self.func_args[self.cur_func] = args

        self.generic_visit(node)
        has_return = any(isinstance(stmt, ast.Return) for stmt in node.body)
        
        return_value = " ^ ".join([f"({arg} = {{{arg}}})" for arg in args])
        return_expr = ast.parse(f'f"{return_value}"').body[0].value

        if has_return:
            for i, stmt in enumerate(node.body):
                if isinstance(stmt, ast.Return):
                    node.body[i].value = return_expr
        else:
            node.body.append(ast.Return(value=return_expr))
        
        return node

    def visit_Return(self, node):
        args = self.func_args[self.cur_func]
        return_value = " ^ ".join([f"({arg} = {{{arg}}})" for arg in args])
        if isinstance(node.value, ast.Constant) and not isinstance(node.value.value, int) and "ERROR" in node.value.value:
            return node
        return_expr = ast.parse(f'f"{return_value}"').body[0].value
        new_node = ast.Return(value=return_expr)
        return ast.copy_location(new_node, node)

def add_return_to_end(code):
    tree = ast.parse(code)
    transformed_tree = AddReturn().visit(tree)
    return ast.unparse(transformed_tree)
