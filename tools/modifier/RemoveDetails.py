import libcst as cst

class RemoveDocstringsCommentsImports(cst.CSTTransformer):
    def leave_Module(self, original_node, updated_node):
        if isinstance(updated_node.body[0], cst.SimpleStatementLine) and isinstance(updated_node.body[0].body[0], cst.Expr) and isinstance(updated_node.body[0].body[0].value, cst.SimpleString):
            updated_body = updated_node.body[1:]
        else:
            updated_body = updated_node.body

        return updated_node.with_changes(body=updated_body)

    def leave_FunctionDef(self, original_node, updated_node):
        if updated_node.body.body and isinstance(updated_node.body.body[0], cst.SimpleStatementLine) and isinstance(updated_node.body.body[0].body[0], cst.Expr) and isinstance(updated_node.body.body[0].body[0].value, cst.SimpleString):
            updated_body = updated_node.body.with_changes(body=updated_node.body.body[1:])
        else:
            updated_body = updated_node.body

        return updated_node.with_changes(body=updated_body)

    def leave_ClassDef(self, original_node, updated_node):
        if updated_node.body.body and isinstance(updated_node.body.body[0], cst.SimpleStatementLine) and isinstance(updated_node.body.body[0].body[0], cst.Expr) and isinstance(updated_node.body.body[0].body[0].value, cst.SimpleString):
            updated_body = updated_node.body.with_changes(body=updated_node.body.body[1:])
        else:
            updated_body = updated_node.body
        return updated_node.with_changes(body=updated_body)

    def leave_Import(self, original_node, updated_node):
        return cst.RemoveFromParent()

    def leave_ImportFrom(self, original_node, updated_node):
        return cst.RemoveFromParent()

    def leave_Comment(self, original_node, updated_node):
        return cst.RemoveFromParent()


class RemoveFunctionDetails(cst.CSTTransformer):
    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        return updated_node.with_changes(returns=None, decorators=[])

    def leave_Parameters(
        self, original_node: cst.Parameters, updated_node: cst.Parameters
    ) -> cst.Parameters:
        new_params = [
            cst.Param(name=param.name, default=None, annotation=None)
            for param in original_node.params
        ]
        return updated_node.with_changes(params=new_params, star_arg=None, star_kwarg=None)
    

def remove_details(source):
    tree = cst.parse_module(source)
    tree = tree.visit(RemoveDocstringsCommentsImports())
    tree = tree.visit(RemoveFunctionDetails())
    return tree.code
