import libcst as cst

class ReplaceCallAndList(cst.CSTTransformer):
    def __init__(self):
        self.strings = []
        self.list_count = 0
        self.listcomp = 0
        self.dict_count = 0
        self.tuple_count = 0
        self.binop_count = 0
        self.unique_node = []
        self.call_name = []
        self.del_mark = "delete_mark"

    def leave_Del(self, original_node: cst.Del, updated_node: cst.Del) -> cst.CSTNode:
        self.call_name.append(self.del_mark)
        return cst.Expr(value=cst.Name(self.del_mark))

    def leave_Expr(self, original_node, updated_node):
        if isinstance(original_node.value, cst.Call):
            func = original_node.value.func
            if isinstance(func, cst.Name):
                return cst.RemovalSentinel.REMOVE
            elif isinstance(func, cst.Attribute):
                nested_func = func
                while isinstance(nested_func, cst.Attribute):
                    nested_func = nested_func.value
                if isinstance(nested_func, cst.Name) and nested_func.value[0].isupper():
                    return cst.RemovalSentinel.REMOVE
        return updated_node

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.CSTNode:
        func = updated_node.func
        if isinstance(func, cst.Name):
            fname = "call_" + func.value
            self.call_name.append(fname)
            return cst.Name(fname)
        elif isinstance(func, cst.Attribute):
            parts = []
            while isinstance(func, cst.Attribute):
                parts.append(func.attr.value)
                func = func.value
            if isinstance(func, cst.Name):
                parts.append(func.value)
            parts.reverse()
            fname = "call_" + "_".join(parts)
            self.call_name.append(fname)
            new_name = cst.Name(fname)
            return new_name
        return updated_node

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.CSTNode:
        if isinstance(original_node.value, cst.ListComp):
            listcomp_name = f"listcomp_{self.listcomp}"
            self.listcomp += 1
            self.call_name.append(listcomp_name)
            return updated_node.with_changes(value=cst.Name(listcomp_name))
        if isinstance(original_node.value, cst.UnaryOperation) and isinstance(original_node.value.operator, cst.Not):
            operand = original_node.value.expression
            if isinstance(operand, cst.Call):
                func = operand.func
                if isinstance(func, cst.Name):
                    fname = "call_not_" + func.value
                    self.call_name.append(fname)
                    new_value = cst.Name(fname)
                    return updated_node.with_changes(value=new_value)
                elif isinstance(func, cst.Attribute):
                    parts = []
                    while isinstance(func, cst.Attribute):
                        parts.append(func.attr.value)
                        func = func.value
                    if isinstance(func, cst.Name):
                        parts.append(func.value)
                    parts.reverse()
                    fname = "call_not_" + "_".join(parts)
                    self.call_name.append(fname)
                    new_value = cst.Name(fname)
                    return updated_node.with_changes(value=new_value)
        return updated_node
    
    def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.BaseExpression:
        if isinstance(updated_node.value, cst.Name) and isinstance(updated_node.attr, cst.Name):
            new_name = f"{updated_node.value.value}_{updated_node.attr.value}"
            self.call_name.append(new_name)
            return cst.Name(new_name)
        return updated_node


    def leave_Subscript(self, original_node: cst.Subscript, updated_node: cst.Subscript) -> cst.CSTNode:
        if isinstance(updated_node.value, cst.Name):
            if len(self.unique_node) == 0:
                self.unique_node.append(updated_node)    
                self.list_count += 1
            else:
                for node in self.unique_node:
                    if node.deep_equals(updated_node):
                        break
                else:
                    self.unique_node.append(updated_node)
                    self.list_count += 1
            if isinstance(original_node.slice[0].slice, cst.Slice):
                list_name = f"{updated_node.value.value}_list_{self.list_count}"
            elif isinstance(original_node.slice[0].slice.value, cst.SimpleString):
                list_name = f"{updated_node.value.value}_dict_{self.list_count}"
            else:
                list_name = f"{updated_node.value.value}_list_{self.list_count}"
            if list_name not in self.call_name:
                self.call_name.append(list_name)
            new_name = cst.Name(value=list_name)
            return new_name
        return updated_node

    def leave_List(self, original_node: cst.List, updated_node: cst.List) -> cst.CSTNode:
        list_name = f"list_{self.list_count}"
        self.list_count += 1
        self.call_name.append(list_name)
        return cst.Name(list_name)
    
    def leave_Tuple(self, original_node: cst.Tuple, updated_node: cst.Tuple) -> cst.CSTNode:
        tuple_name = f"tuple_{self.tuple_count}"
        self.tuple_count += 1
        self.call_name.append(tuple_name)
        return cst.Name(tuple_name)
    
    def leave_Dict(self, original_node: cst.Dict, updated_node: cst.Dict) -> cst.CSTNode:
        dict_name = f"dict_{self.dict_count}"
        self.dict_count += 1
        self.call_name.append(dict_name)
        return cst.Name(dict_name)

    def leave_Comparison(self, original_node: cst.Comparison, updated_node: cst.Comparison) -> cst.BaseExpression:
        if any(isinstance(op.operator, (cst.In, cst.NotIn)) for op in updated_node.comparisons):
            count = 1
            if isinstance(updated_node.left, cst.SimpleString):
                if updated_node.left.value in self.strings:
                    nstr = self.strings.index(updated_node.left.value)
                else:
                    nstr = len(self.strings)
                    self.strings.append(updated_node.left.value)
                new_param = f"str{nstr}_in_list"
            elif isinstance(original_node.left, cst.Integer):
                new_param = f"num_{updated_node.left.value}_in_list"
            elif isinstance(original_node.left, cst.UnaryOperation):
                if isinstance(original_node.left.expression, cst.Integer):
                    new_param = f"num__{updated_node.left.expression.value}_in_list"
            elif isinstance(updated_node.left, cst.BinaryOperation):
                new_param = f"bin_op_{self.binop_count}_in_list"
            elif isinstance(updated_node.left, cst.Tuple):
                new_param = f"tuple_{self.tuple_count}_in_list"
            else:
                new_param = f"{updated_node.left.value}_in_list"
            self.call_name.append(new_param)
            new_left = cst.Name(value=new_param)
            return new_left
        return updated_node


def replace_call(source):
    tree = cst.parse_module(source)
    transformer = ReplaceCallAndList()
    tree = tree.visit(transformer)
    return tree.code, transformer.call_name
