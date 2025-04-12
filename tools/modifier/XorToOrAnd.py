import libcst as cst
import libcst.matchers as m


class XorToOrAnd(cst.CSTTransformer):
    def __init__(self):
        self.removed_assignments = []

    def leave_If(self, original_node, updated_node):
        if m.matches(updated_node.test, m.BinaryOperation(left=m.Name(), operator=m.BitXor(), right=m.Name())):
            left = updated_node.test.left
            right = updated_node.test.right
            new_test = cst.BooleanOperation(
                left=cst.UnaryOperation(operator=cst.Not(), expression=left),
                operator=cst.And(),
                right=right
            )
            new_test = cst.BooleanOperation(
                left=new_test,
                operator=cst.Or(),
                right=cst.BooleanOperation(
                    left=left,
                    operator=cst.And(),
                    right=cst.UnaryOperation(operator=cst.Not(), expression=right)
                )
            )
            return updated_node.with_changes(test=new_test)

        if m.matches(updated_node.test, m.UnaryOperation(operator=cst.Not(), expression=m.BinaryOperation(left=m.Name(), operator=m.BitXor(), right=m.Name()))):
            left = updated_node.test.expression.left
            right = updated_node.test.expression.right
            new_test = cst.BooleanOperation(
                left=cst.UnaryOperation(operator=cst.Not(), expression=left),
                operator=cst.And(),
                right=right
            )
            new_test = cst.BooleanOperation(
                left=new_test,
                operator=cst.Or(),
                right=cst.BooleanOperation(
                    left=left,
                    operator=cst.And(),
                    right=cst.UnaryOperation(operator=cst.Not(), expression=right)
                )
            )
            new_test = cst.UnaryOperation(operator=cst.Not(), expression=cst.BooleanOperation(
                left=new_test,
                operator=cst.Or(),
                right=cst.BooleanOperation(
                    left=left,
                    operator=cst.And(),
                    right=cst.UnaryOperation(operator=cst.Not(), expression=right)
                ),
                lpar=[cst.LeftParen()],
                rpar=[cst.RightParen()]
            ))
            return updated_node.with_changes(test=new_test)

        return updated_node

    def leave_Assign(self, original_node, updated_node):
        if (m.matches(updated_node.value, m.BinaryOperation(left=m.Name(), operator=m.BitXor(), right=m.Name()))):
            target = updated_node.targets[0].target
            if isinstance(target, cst.Name):
                self.removed_assignments.append(target.value)
            return cst.RemoveFromParent()

        return updated_node

    

def modify_xor(source_code):
    module = cst.parse_module(source_code)
    transformer = XorToOrAnd()
    transformed_module = module.visit(transformer)
    transformed_code = transformed_module.code
    return transformed_code, transformer.removed_assignments
