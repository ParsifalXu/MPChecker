import ast

class TernaryTransformer(ast.NodeTransformer):
    def visit_Assign(self, node):
        if isinstance(node.value, ast.IfExp):
            target = node.targets[0]
            test = node.value.test
            body = node.value.body
            orelse = node.value.orelse

            if_node = ast.If(
                test=test,
                body=[ast.Assign(targets=[target], value=body)],
                orelse=[ast.Assign(targets=[target], value=orelse)]
            )
            return if_node
        return node


def modify_ternary(source):
    tree = ast.parse(source)
    modified_tree = TernaryTransformer().visit(tree)
    modified_tree = ast.fix_missing_locations(modified_tree)
    return ast.unparse(modified_tree)

