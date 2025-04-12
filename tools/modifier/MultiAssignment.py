import libcst as cst

class multiAssignment(cst.CSTTransformer):
    def leave_Assign(self, original_node, updated_node):
        if len(updated_node.targets) == 1 and isinstance(updated_node.targets[0].target, cst.Tuple):
            target_tuple = updated_node.targets[0].target
            value = updated_node.value
            new_assignments = [
                cst.Assign(
                    targets=[cst.AssignTarget(target=element.value)],
                    value=value
                ) for element in target_tuple.elements
            ]
            return cst.FlattenSentinel(new_assignments)

        return updated_node
    
def reconstuct_multiple_assignment(source):
    tree = cst.parse_module(source)
    tree = tree.visit(multiAssignment())
    return tree.code