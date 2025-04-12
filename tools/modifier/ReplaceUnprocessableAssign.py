import libcst as cst

class ReplaceWarningAndError(cst.CSTTransformer):
    def leave_Raise(self, original_node: cst.Raise, updated_node: cst.Raise) -> cst.CSTNode:
        return cst.Return(value=cst.SimpleString('"ERROR_END"'))
    
    def leave_Assert(self, original_node: cst.Assert, updated_node: cst.Assert) -> cst.CSTNode:
        return cst.Return(value=cst.SimpleString('"ASSERT_END"'))
    
    def leave_FormattedString(self, original_node: cst.FormattedString, updated_node: cst.FormattedString) -> cst.CSTNode:
        return cst.FormattedString(parts=[cst.FormattedStringText(value="String Content")])

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.CSTNode:
        if isinstance(updated_node.func, cst.Attribute) and isinstance(updated_node.func.value, cst.Name):
            if updated_node.func.value.value == "warnings" and updated_node.func.attr.value == "warn":
                return cst.Return(value=cst.SimpleString('"WARNING_END"'))
        return updated_node
    
class ReplaceFloat(cst.CSTTransformer):
    def leave_Float(self, original_node: cst.Float, updated_node: cst.Float) -> cst.BaseExpression:
        return cst.Integer(value="1")

class ReplaceMatMulToDot(cst.CSTTransformer):
    def leave_BinaryOperation(self, original_node: cst.BinaryOperation, updated_node: cst.BinaryOperation) -> cst.CSTNode:
        if isinstance(original_node.operator, cst.MatrixMultiply):
            left = updated_node.left
            right = updated_node.right
            new_call = cst.Call(
                func=cst.Attribute(
                    value=cst.Name(value="np"),
                    attr=cst.Name(value="dot")
                ),
                args=[cst.Arg(value=left), cst.Arg(value=right)]
            )
            return new_call
        return updated_node
    
class RemoveSelf(cst.CSTTransformer):
    def __init__(self):
        self.init_param = []

    def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.CSTNode:
        if isinstance(original_node.value, cst.Name) and original_node.value.value == "self":
            self.init_param.append(original_node.attr.value)
            return cst.Name(value=original_node.attr.value)
        return updated_node

def replace_unprocessable_assign(source):
    tree = cst.parse_module(source)
    tree = tree.visit(ReplaceWarningAndError())
    tree = tree.visit(ReplaceFloat())
    tree = tree.visit(ReplaceMatMulToDot())
    rstransformer = RemoveSelf()
    tree = tree.visit(rstransformer)
    return tree.code, rstransformer.init_param
