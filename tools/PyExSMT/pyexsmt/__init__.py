import logging
import re
import sys

from pyexsmt.symbolic_types import SymbolicObject
from pyexsmt.symbolic_types.symbolic_int import SymbolicInteger
from pyexsmt.symbolic_types.symbolic_object import to_pysmt

from pysmt.shortcuts import *

TYPES ={"INT": INT, "BOOL": BOOL, "REAL":REAL}

def pred_to_smt(pred):
    '''
    pred : symbolic.preidcate.Predicate
    returns boolean pySMT expr
    '''
    if pred is None:
        return TRUE()

    if not pred.result:
        ret = Not(pred.symtype.expr)
    else:
        ret = pred.symtype.expr

    logging.debug("PREDICATE: %s", ret)
    return ret

def get_concr_value(variable):
    '''
    variable : (U SymbolicObject PythonPrimitive)
    return a python primitive that is either the input
    or the current concrete value for the symbolic object
    '''
    if isinstance(variable, SymbolicObject):
        return variable.get_concr_value()
    else:
        return variable

def parse_types(type_list):
    '''
    take a list of two strings, type_list.
    type_list[0] is the return type.
    type_list[1] is the argument types.
    something like ["INT", "[INT,INT]"].
    Return the correct type list.
    E.g. [INT, [INT, INT]]
    '''
    try:
        return_type = TYPES[type_list[0].upper()]
        arg_types = re.findall('\w+', type_list[1])
        arg_types = [TYPES[a.upper()] for a in arg_types]
        return [return_type, arg_types]
    except KeyError:
        logging.error("Unsupported types in uniterpreted function definition: %s. \
        \nSupported types are:%s", type_list, list(TYPES.keys()))
        sys.exit(-1)

def uninterp_func_pair(definition, module):
    '''
    definition : [name : String, return_type : String, argument_types : String]
    argument_types matches "[<Type>,<Type>,...]"

    returns a pair where the first element is the concrete function we want to replace
    and the second element is the symbolic function we want to plug in
    '''
    funcs = []
    if not definition is None:
        module_func = module+"."+definition[0]
        func_types = parse_types(definition[1:])
        ftype = FunctionType(*func_types)
        f = Symbol(definition[0], ftype)
        def wrapper(*args):
            '''
            IMPORTANT OR ELSE SYMBOLIC VARIABLES GET CAUGHT IN PROCESSING
            AND WE GET WEIRD PATHS BEYOND MODULE IN QUESTION
            '''
            args = [to_pysmt(a) for a in args]
            try:
                ret = f(*args)
                return get_symbolic_from_expr(ret)
            except Exception:
                logging.error("Failed to call %s of type %s with args %s.", f, ftype, [a.get_type() for a in args])
                sys.exit(-1)
        funcs = [(module_func, wrapper)]
    return funcs

def get_symbolic_from_expr(expr):
    '''
    expr : pySMT Object (FNode)
    return a SymbolicObject that wraps expr
    '''
    if expr.get_type().is_int_type():
        return SymbolicInteger(expr)
    elif expr.get_type().is_bool_type():
        return SymbolicObject(expr)
    else:
        logging.error("TYPE NOT FOUND: %s", expr.get_type())
        sys.exit(-1)

def match_smt_type(node, to_type):
    '''
    expr : pySMT Object (FNode)
    to_type : pySMT TYpe

    return a PySMT Object for node of type to_type
    '''
    if node.get_type() == to_type:
        return node
    else:
        return Symbol(str(node), to_type)
