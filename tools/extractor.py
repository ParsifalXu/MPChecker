import os
import ast

from tools.macros import _DOWNLOAD_DIR
from tools.macros import INFO_DIR
from tools.macros import FLAG

def extract_class_and_independent_function(project):
    project_path = os.path.join(_DOWNLOAD_DIR, project)
    if not os.path.exists(project_path):
        print(f"Please download the project first: {project}")
        exit(0)
    info_path = os.path.join(INFO_DIR, project)
    if not os.path.exists(info_path):
        os.makedirs(info_path)
    
    codefiles = []
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py") and "{" not in file:
                codefiles.append(os.path.join(root, file))

    statistics = {
        "class": 0,
        "class_with_docstring": 0,
        "inside_function": 0,
        "inside_function_with_docstring": 0,
        "independent_function": 0,
        "independent_function_with_docstring": 0
    }
    
    for file in codefiles:
        statistics = extract_content(file, project, statistics)
    
    for root, dirs, files in os.walk(info_path, topdown=False):
        for name in dirs:
            folder_path = os.path.join(root, name)
            if not os.listdir(folder_path):
                os.rmdir(folder_path)
    
    print(f"========== PROJECT: {project} ==========")
    print(f"CLASS: {statistics['class']}")
    print(f"CLASS_WITH_DOCSTRING: {statistics['class_with_docstring']}")
    print(f"INSIDE FUNCTION: {statistics['inside_function']}")
    print(f"INSIDE FUNCTION WITH DOCSTRING: {statistics['inside_function_with_docstring']}")
    print(f"INDEPENDENT FUNCTION: {statistics['independent_function']}")
    print(f"INDEPENDENT FUNCTION WITH DOCSTRING: {statistics['independent_function_with_docstring']}")
    print(f"=============== END ===============")

def extract_content(file, project, statistics):
    with open(file, 'r') as f:
        content = f.read()
    f.close()

    flag = FLAG[project]
    if flag == "Parameters":
        flag = flag + "\n-----"
    
    class ClassAndFunctionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.classes = []
            self.classes_with_docstring = []
            self.inside_functions = []
            self.inside_functions_with_docstring = []
            self.independent_functions = []
            self.independent_functions_with_docstring = []

        def visit_ClassDef(self, node):
            self.classes.append(node.name)
            docstring = ast.get_docstring(node)
            if docstring and flag in docstring:
                node.decorator_list = []
                self.classes_with_docstring.append(node.name)
                if not os.path.exists(f'{INFO_DIR}/{project}/{node.name}'):
                    os.makedirs(f'{INFO_DIR}/{project}/{node.name}')
                with open(f'{INFO_DIR}/{project}/{node.name}/{node.name}.py', 'w') as fc:
                    fc.write(ast.unparse(node))
                with open(f'{INFO_DIR}/{project}/{node.name}/{node.name}_docstring.txt', 'w') as fd:
                    fd.write(docstring)
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            if not isinstance(node.parent, ast.ClassDef):
                self.independent_functions.append(node.name)
                docstring = ast.get_docstring(node)
                if docstring and flag in docstring:
                    node.decorator_list = []
                    self.independent_functions_with_docstring.append(node.name)
                    if not os.path.exists(f'{INFO_DIR}/{project}/{node.name}'):
                        os.makedirs(f'{INFO_DIR}/{project}/{node.name}')
                    with open(f'{INFO_DIR}/{project}/{node.name}/{node.name}.py', 'w') as fc:
                        fc.write(ast.unparse(node))
                    with open(f'{INFO_DIR}/{project}/{node.name}/{node.name}_docstring.txt', 'w') as fd:
                        fd.write(docstring)
            else:
                if node.name != "__init__":
                    node.decorator_list = []
                    self.inside_functions.append(node.name)
                    if not os.path.exists(f'{INFO_DIR}/{project}/{node.parent.name}/memberfunc/{node.name}'):
                        os.makedirs(f'{INFO_DIR}/{project}/{node.parent.name}/memberfunc/{node.name}')
                    with open(f'{INFO_DIR}/{project}/{node.parent.name}/memberfunc/{node.name}/{node.name}.py', 'w') as fc:
                        fc.write(ast.unparse(node))
                    docstring = ast.get_docstring(node)
                    if docstring and flag in docstring:
                        self.inside_functions_with_docstring.append(node.name)
                        with open(f'{INFO_DIR}/{project}/{node.parent.name}/memberfunc/{node.name}/{node.name}_docstring.txt', 'w') as fd:
                            fd.write(docstring)
            self.generic_visit(node) 

    def add_parent_info(node):
        for child in ast.iter_child_nodes(node):
            child.parent = node
            add_parent_info(child)

    tree = ast.parse(content)
    add_parent_info(tree)
    visitor = ClassAndFunctionVisitor()
    visitor.visit(tree)

    statistics['class'] = statistics['class'] + len(visitor.classes)
    statistics['class_with_docstring'] = statistics['class_with_docstring'] + len(visitor.classes_with_docstring)
    statistics['inside_function'] = statistics['inside_function'] + len(visitor.inside_functions)
    statistics['inside_function_with_docstring'] = statistics['inside_function_with_docstring'] + len(visitor.inside_functions_with_docstring)
    statistics['independent_function'] = statistics['independent_function'] + len(visitor.independent_functions)
    statistics['independent_function_with_docstring'] = statistics['independent_function_with_docstring'] + len(visitor.independent_functions_with_docstring)

    return statistics