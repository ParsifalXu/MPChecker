# Copyright: see copyright.txt

from pyexsmt.symbolic_types.symbolic_int import SymbolicInteger
from pyexsmt.symbolic_types.symbolic_int import SymbolicObject

def get_symbolic(v):
    exported = [(int, SymbolicInteger), (bool, SymbolicObject)]
    for (t, s) in exported:
        if isinstance(v, t):
            return s
    return None
