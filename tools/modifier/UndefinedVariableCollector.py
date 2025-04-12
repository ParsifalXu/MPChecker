import ast

class UndefinedVariableFinder(ast.NodeVisitor):
    def __init__(self):
        self.defined_vars = set()  
        self.used_vars = set()     
        self.undefined_vars = set()

    def visit_FunctionDef(self, node):
        self.defined_vars = {arg.arg for arg in node.args.args}
        self.used_vars = set()
        self.generic_visit(node)
        self.undefined_vars = self.used_vars - self.defined_vars

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.defined_vars.add(target.id)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used_vars.add(node.id)

    def find_undefined_vars(self, source_code):
        tree = ast.parse(source_code)
        self.visit(tree)
        return self.undefined_vars
    
def undef_var_collector(source_code):
    finder = UndefinedVariableFinder()
    undefined_vars = finder.find_undefined_vars(source_code)
    return undefined_vars