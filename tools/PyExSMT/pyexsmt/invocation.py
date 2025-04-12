# Copyright: see copyright.txt

from contextlib import ExitStack
from mock import patch

class FunctionInvocation:
    def __init__(self, function, reset):
        self.function = function
        self.reset = reset
        self.arg_constructor = {}

    def call_function(self, args, funcs=[]):
        '''
        funcs should be a list of pairs. 
        Left element is a concrete function
        right element is the symbolic replacement
        '''
        self.reset()
        with ExitStack() as stack:
            [stack.enter_context(patch(cf, lf, create=True)) for cf, lf in funcs]
            ret = self.function(**args)
        return ret

    def add_arg_constructor(self, name, constructor):
        self.arg_constructor[name] = constructor

    def get_names(self):
        return self.arg_constructor.keys()

    def create_arg_value(self, name):
        return self.arg_constructor[name](name)
