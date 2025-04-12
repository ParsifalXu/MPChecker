import libcst as cst

class FunctionParamExtractor(cst.CSTVisitor):
    def __init__(self):
        self.params = []

    def visit_FunctionDef(self, node: cst.FunctionDef):
        self.params = [
            param.name.value for param in node.params.params if param.name.value != "self"
        ]
        return False

def extract_function_params(source_code):
    tree = cst.parse_module(source_code)
    extractor = FunctionParamExtractor()
    tree.visit(extractor)
    return tree.code, extractor.params