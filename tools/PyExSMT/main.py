import os
import sys
import logging
import random
from argparse import ArgumentParser

from pyexsmt import uninterp_func_pair
from pyexsmt.loader import *
from pyexsmt.explore import ExplorationEngine

from pysmt.shortcuts import *

def main():
    sys.path = [os.path.abspath(os.path.join(os.path.dirname(__file__)))] + sys.path
    parser = ArgumentParser()

    parser.add_argument("--log", dest="loglevel", action="store", \
                                    help="Set log level", default="")
    parser.add_argument('--uninterp', dest="uninterp", nargs=3, \
                                    metavar=('name', 'return_type', 'arg_types'), \
                                    help='<func_name> <return_type> <arg_types>')
    parser.add_argument("--entry", dest="entry", action="store", \
                                    help="Specify entry point", default="")
    parser.add_argument("--graph", dest="dot_graph", action="store_true", \
                                    help="Generate a DOT graph of execution tree")
    parser.add_argument("--path", dest="path", action="store_true", \
                                    help="Find all paths of execution tree")
    parser.add_argument("--summary", dest="summary", action="store_true", \
                                    help="Generate a functional summary")
    parser.add_argument("--max-iters", dest="max_iters", type=int, \
                                    help="Limit number of iterations", default=0)
    parser.add_argument("--max-depth", dest="max_depth", type=int, \
                                    help="Limit the depth of paths", default=0)
    parser.add_argument("--solver", dest="solver", action="store", \
                                    help="Choose SMT solver", default="z3")
    parser.add_argument(dest="file", action="store", help="Select Python file")
    options = parser.parse_args()

    if options.loglevel in ["info", "INFO", "i", "I"]:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
    elif options.loglevel in ["debug", "DEBUG", "d", "D"]:
        logging.basicConfig(level=logging.DEBUG, format='DEBUG:\t%(message)s')
    elif options.loglevel == "":
        pass
    else:
        logging.error("Unrecognized Log Level")
        sys.exit(-1)

    logging.debug("Log Level Set to Debug")

    if options.file == "" or not os.path.exists(options.file):
        parser.error("Missing app to execute")
        sys.exit(1)
    mapping = replace_str2num(options.file)

    if not options.solver in get_env().factory.all_solvers():
        logging.error("Solver %s not available", options.solver)
        sys.exit(-1)
    else:
        solver = options.solver

    summary = options.summary
    filename = os.path.abspath(options.file)
    app = loaderFactory(filename, options.entry)
    if app is None:
        sys.exit(1)

    funcs = uninterp_func_pair(options.uninterp, app.get_file())
    result = None
    try:
        engine = ExplorationEngine(app.create_invocation(), solver=solver)
        result_struct = engine.explore(options.max_iters, options.max_depth, funcs)
        return_vals = result_struct.execution_return_values
        result = app.execution_complete(return_vals)

        if summary:
            summary = result_struct.to_summary()
            print("\nSummary:\n%s\n" % summary)

        if options.dot_graph:
            result_struct.to_dot(filename, mapping)

        if options.path:
            result_struct.to_path(filename, mapping)

        replace_num2str(options.file, mapping)

    except (ImportError, NotImplementedError, TypeError) as error:
        logging.error(error)
        sys.exit(1)
    if result is None or result:
        sys.exit(0)
    else:
        sys.exit(1)


def replace_str2num(file):
    with open(file, 'r') as f:
        content = f.read()
    strings = re.findall(r"'(.*?)'", content)
    strings = list(set(strings))
    mapping = {}

    for string in strings:
        random_number = random.randint(10000, 99999)
        if random_number in mapping.values():
            random_number = random.randint(10000, 99999)
        mapping[string] = random_number
        content = content.replace(f"'{string}'", str(random_number))

    with open(file, 'w') as f:
        f.write(content)

    return mapping

def replace_num2str(file, mapping):
    with open(file, 'r') as f:
        content = f.read()  
    for key, value in mapping.items():
        content = content.replace(str(value), f"'{key}'")
    with open(file, 'w') as f:
        f.write(content)



if __name__ == "__main__":
    main()