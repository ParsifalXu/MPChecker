import libcst as cst

class LoopToIfTransformer(cst.CSTTransformer):
    def __init__(self):
        self.loop_count = 1
        self.loopname = []

    def leave_Break(self, original_node, updated_node):
        return cst.Pass()
    
    def leave_Continue(self, original_node, updated_node):
        return cst.Pass()

    def leave_For(self, original_node, updated_node):
        if (
            isinstance(updated_node.iter, cst.Call)
            and isinstance(updated_node.iter.func, cst.Name)
            and updated_node.iter.func.value == "range"
            and len(updated_node.iter.args) == 1
            and isinstance(updated_node.iter.args[0].value, cst.Integer)
            and updated_node.iter.args[0].value.value == "1"
        ):
            loop_var_name = f"loop{self.loop_count}"
            self.loopname.append(loop_var_name)
            self.loop_count += 1

            new_if_node = cst.If(
                test=cst.Comparison(
                    left=cst.Name(loop_var_name),
                    comparisons=[cst.ComparisonTarget(operator=cst.Equal(), comparator=cst.Integer("1"))],
                ),
                body=updated_node.body,
                orelse=None,
            )
            return new_if_node
        return updated_node

    def leave_While(self, original_node, updated_node):
        if isinstance(updated_node.test, cst.Name) and updated_node.test.value == "True":
            loop_var_name = f"loop{self.loop_count}"
            self.loopname.append(loop_var_name)
            self.loop_count += 1

            new_if_node = cst.If(
                test=cst.Comparison(
                    left=cst.Name(loop_var_name),
                    comparisons=[cst.ComparisonTarget(operator=cst.Equal(), comparator=cst.Integer("1"))],
                ),
                body=updated_node.body,
                orelse=None,
            )
            return new_if_node
        return updated_node

def replace_loop(code):
    tree = cst.parse_module(code)
    transformer = LoopToIfTransformer()
    transformed_module = tree.visit(transformer)
    return transformed_module.code, transformer.loopname
