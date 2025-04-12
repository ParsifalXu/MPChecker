import libcst as cst

class ReplaceIsToEqualAndStrify(cst.CSTTransformer):
    def leave_Comparison(self, original_node: cst.Comparison, updated_node: cst.Comparison) -> cst.CSTNode:
        new_comparisons = []
        for comparison in updated_node.comparisons:
            if isinstance(comparison.operator, cst.IsNot):
                new_comparisons.append(comparison.with_changes(operator=cst.NotEqual()))
            elif isinstance(comparison.operator, cst.Is):
                new_comparisons.append(comparison.with_changes(operator=cst.Equal()))
            else:
                new_comparisons.append(comparison)
        return updated_node.with_changes(comparisons=new_comparisons)

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.CSTNode:
        if updated_node.value in ["True", "False", "None"]:
            return cst.SimpleString(f'"{updated_node.value}"')
        return updated_node

class ReplaceNotToEqualsZero(cst.CSTTransformer):
    def leave_UnaryOperation(self, original_node: cst.UnaryOperation, updated_node: cst.UnaryOperation) -> cst.CSTNode:
        if isinstance(updated_node.operator, cst.Not):
            return cst.Comparison(
                left=updated_node.expression,
                comparisons=[
                    cst.ComparisonTarget(
                        operator=cst.Equal(),
                        comparator=cst.Integer("0")
                    )
                ]
            )
        return updated_node
    
class SimplifyForLoop(cst.CSTTransformer):
    def leave_For(self, original_node: cst.For, updated_node: cst.For) -> cst.For:
        new_for_loop = cst.For(
            target=cst.Name("i"),
            iter=cst.Call(func=cst.Name("range"), args=[cst.Arg(cst.Integer("1"))]),
            body=updated_node.body,
            orelse=updated_node.orelse
        )
        return new_for_loop

def replace_comparison_operators(code):
    tree = cst.parse_module(code)
    tree = tree.visit(ReplaceIsToEqualAndStrify())
    tree = tree.visit(ReplaceNotToEqualsZero())
    tree = tree.visit(SimplifyForLoop())
    return tree.code
