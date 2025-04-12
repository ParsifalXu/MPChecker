# Copyright: copyright.txt

import logging
import inspect
import re
import os
import sys
import builtins
from pyexsmt.invocation import FunctionInvocation
from pyexsmt.symbolic_types import SymbolicInteger, get_symbolic

# The built-in definition of len wraps the return value in an int()
# constructor, destroying any symbolic types.
# By redefining len here we can preserve symbolic integer types.
builtins.len = (lambda x: x.__len__())

class Loader:
    def __init__(self, filename, entry):
        self._file_name = os.path.basename(filename)
        self._file_name = self._file_name[:-3]
        if (entry == ""):
            self._entry_point = self._file_name
        else:
            self._entry_point = entry
        self._reset_callback(True)

    def get_file(self):
        return self._file_name

    def get_entry(self):
        return self._entry_point
    
    def create_invocation(self):
        inv = FunctionInvocation(self._execute,self._reset_callback)
        func = self.app.__dict__[self._entry_point]
        argspec = inspect.signature(func)
        # check to see if user specified initial values of arguments
        if "concrete_args" in func.__dict__:
            for (f,v) in func.concrete_args.items():
                if not f in argspec.parameters:
                    raise ImportError("Error in @concrete: " +  self._entry_point + " has no argument named " + f)
                else:
                    Loader._init_arg_concrete(inv,f,v)
        if "symbolic_args" in func.__dict__:
            for (f,v) in func.symbolic_args.items():
                if not f in argspec.parameters:
                    raise ImportError("Error (@symbolic): " +  self._entry_point + " has no argument named " + f)
                elif f in inv.get_names():
                    raise ImportError("Argument " + f + " defined in both @concrete and @symbolic")
                else:
                    s = get_symbolic(v)
                    if (s == None):
                        raise ImportError("Error at argument " + f + " of entry point " + self._entry_point + " : no corresponding symbolic type found for type " + str(type(v)))
                    Loader._init_arg_symbolic(inv, f, s)
        for a in argspec.parameters:
            if not a in inv.get_names():
                Loader._init_arg_symbolic(inv, a, SymbolicInteger)
        return inv

    # need these here (rather than inline above) to correctly capture values in lambda
    def _init_arg_concrete(inv, f, val):
        inv.add_arg_constructor(f, lambda n: val)

    def _init_arg_symbolic(inv, f, st):
        inv.add_arg_constructor(f, lambda n: st(None, n))

    def execution_complete(self, return_vals):
        if "expected_result" in self.app.__dict__:
            return self._check(return_vals, self.app.__dict__["expected_result"]())
        if "expected_result_set" in self.app.__dict__:
            return self._check(return_vals, self.app.__dict__["expected_result_set"](),False)
        else:
            logging.info(self._file_name + ".py contains no expected_result function")
            return None

    # -- private

    def _reset_callback(self,firstpass=False):
        self.app = None
        if firstpass and self._file_name in sys.modules:
            raise ImportError("There already is a module loaded named " + self._file_name)
        try:
            if not firstpass and self._file_name in sys.modules:
                del sys.modules[self._file_name]
            self.app = __import__(self._file_name)
            if not self._entry_point in self.app.__dict__ or not callable(self.app.__dict__[self._entry_point]):
                raise ImportError("File %s.py doesn't contain a function named %s"
                                  % (self._file_name, self._entry_point))
        except Exception as arg:
            raise ImportError("Couldn't import " + self._file_name + "\n" + arg)

    def _execute(self, **args):
        return self.app.__dict__[self._entry_point](**args)

    def _check(self, computed, expected, as_bag=True):
        b_c = _to_bag(computed)
        b_e = _to_bag(expected)
        if as_bag and b_c != b_e or not as_bag and set(computed) != set(expected):
            print("-------------------> %s Test Failed <---------------------" % self._file_name)
            print("Expected: %s, found: %s" % (b_e, b_c))
            return False
        else:
            print("Test Passed <--- %s" % self._file_name)
            return True

def loaderFactory(filename, entry):
    if not os.path.isfile(filename) or not re.search(".py$", filename):
        print("Please provide a Python file to load")
        return None
    try:
        directory = os.path.dirname(filename)
        sys.path = [directory] + sys.path
        ret = Loader(filename, entry)
        return ret
    except ImportError:
        sys.path = sys.path[1:]
        return None

def _to_bag(l):
    bag = {}
    for i in l:
        if i in bag:
            bag[i] += 1
        else:
            bag[i] = 1
    return bag
