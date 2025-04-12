# Copyright: see copyright.txt

import logging
import inspect

from pysmt.shortcuts import *

# the ABSTRACT base class for representing any expression that depends on a symbolic input
# it also tracks the corresponding concrete value for the expression (aka concolic execution)

class SymbolicObject(object):
    def __init__(self, expr, name="se", ty=INT):
        if expr is None:
            self.expr = Symbol(name, ty)
        else:
            self.expr = expr

    # This is set up by the concolic engine to link __bool__ to PathConstraint
    SI = None
    # This is set up by the concolic engine to link the solver to the variables
    SOLVER = None

    # this is a critical interception point: the __bool__
    # method is called whenever a predicate is evaluated in
    # Python execution (if, while, and, or). This allows us
    # to capture the path condition
    def __bool__(self):
        t = self.expr.get_type()
        if t == BOOL:
            obj = self
        elif t == INT:
            obj = SymbolicObject(NotEquals(self.expr, Int(0)))
        else:
            raise NotImplementedError("%s not supported in conditional yet" % t)

        ret = obj.get_concr_value()

        if SymbolicObject.SI != None:
            SymbolicObject.SI.which_branch(ret, obj)

        return ret

    def get_concr_value(self):
        if SymbolicObject.SOLVER is None:
            raise ValueError("MUST SPECIFY SOLVER")
        if not SymbolicObject.SOLVER.last_result:
            raise ValueError("SOLVER MUST HAVE A MODEL")
        val = SymbolicObject.SOLVER.get_py_value(self.expr)
        return val

    def symbolic_eq(self, other):
        if not isinstance(other, SymbolicObject):
            ret = False
        ret = repr(self.expr) == repr(other.expr)
        logging.debug("Checking equality of %s and %s: result is %s", repr(self), repr(other), ret)
        return ret

    def __hash__(self):
        return hash(self.get_concr_value())

    def __str__(self):
        return str(self.get_concr_value())

    def __repr__(self):
        return self.expr.serialize()


    ## COMPARISON OPERATORS
    def __eq__(self, other):
        #TODO: what if self is not symbolic and other is?
        other = to_pysmt(other)
        if self.expr.get_type() != other.get_type():
            return False
        return SymbolicObject(Equals(self.expr, other))

    def __ne__(self, other):
        other = to_pysmt(other)
        if self.expr.get_type() != other.get_type():
            return False
        return SymbolicObject(NotEquals(self.expr, other))

    def __lt__(self, other):
        other = to_pysmt(other)
        if self.expr.get_type() != other.get_type():
            return False
        return SymbolicObject(LT(self.expr, other))

    def __le__(self, other):
        other = to_pysmt(other)
        if self.expr.get_type() != other.get_type():
            return False
        return SymbolicObject(LE(self.expr, other))

    def __gt__(self, other):
        other = to_pysmt(other)
        if self.expr.get_type() != other.get_type():
            return False
        return SymbolicObject(GT(self.expr, other))

    def __ge__(self, other):
        other = to_pysmt(other)
        if self.expr.get_type() != other.get_type():
            return False
        return SymbolicObject(GE(self.expr, other))

    ## LOGICAL OPERATORS
    def __and__(self, other):
        other = to_pysmt(other)
        if self.expr.get_type() != other.get_type():
            raise TypeError("CANNOT AND %s and %s" %(self.expr.get_type(), other.get_type()))
        return SymbolicObject(And(self.expr, other))

    def __or__(self, other):
        other = to_pysmt(other)
        if self.expr.get_type() != other.get_type():
            raise TypeError("CANNOT OR %s and %s" %(self.expr.get_type(), other.get_type()))
        return SymbolicObject(Or(self.expr, other))

    ## UNARY OPERATORS
    def __neg__(self):
        raise NotImplementedError("neg is not implemented for %s" % self.expr.get_type())

    def __pos__(self):
        raise NotImplementedError("pos is not implemented for %s" % self.expr.get_type())

    def __abs__(self):
        raise NotImplementedError("abs is not implemented for %s" % self.expr.get_type())

    ## ARITHMETIC OPERATORS
    def __add__(self, other):
        raise NotImplementedError("add is not implemented for %s!" % self.expr.get_type())

    def __sub__(self, other):
        raise NotImplementedError("sub is not implemented for %s!" % self.expr.get_type())

    def __mul__(self, other):
        raise NotImplementedError("mul is not implemented for %s!" % self.expr.get_type())

    def __mod__(self, other):
        raise NotImplementedError("mod is not implemented for %s!" % self.expr.get_type())

    def __div__(self, other):
        raise NotImplementedError("div is not implemented for %s!" % self.expr.get_type())

    def __floordiv__(self, other):
        raise NotImplementedError("floordiv is not implemented for %s!" % self.expr.get_type())

    def __truediv__(self, other):
        raise NotImplementedError("truediv is not implemented for %s!" % self.expr.get_type())

    def __divmod__(self, other):
        raise NotImplementedError("divmod is not implemented for %s!" % self.expr.get_type())

    def __pow__(self, other):
        raise NotImplementedError("pow is not implemented for %s!" % self.expr.get_type())

    def __xor__(self, other):
        raise NotImplementedError("xor is not implemented for %s!" % self.expr.get_type())

    def __lshift__(self, other):
        raise NotImplementedError("lshift is not implemented for %s!" % self.expr.get_type())

    def __rshift__(self, other):
        raise NotImplementedError("rshift is not implemented for %s!" % self.expr.get_type())

    ## REVERSE OPERATORS
    def __radd__(self, other):
        raise NotImplementedError("radd is not implemented for %s!" % self.expr.get_type())

    def __rsub__(self, other):
        raise NotImplementedError("rsub is not implemented for %s!" % self.expr.get_type())

    def __rmul__(self, other):
        raise NotImplementedError("rmul is not implemented for %s!" % self.expr.get_type())

    def __rdiv__(self, other):
        raise NotImplementedError("rdiv is not implemented for %s!" % self.expr.get_type())

    def __rfloordiv__(self, other):
        raise NotImplementedError("rfloordiv is not implemented for %s!" % self.expr.get_type())

    def __rtruediv__(self, other):
        raise NotImplementedError("rtruediv is not implemented for %s!" % self.expr.get_type())

    def __rmod__(self, other):
        raise NotImplementedError("rmod is not implemented for %s!" % self.expr.get_type())

    def __rdivmod__(self, other):
        raise NotImplementedError("rdivmod is not implemented for %s!" % self.expr.get_type())

    def __rpow__(self, other):
        raise NotImplementedError("rpow is not implemented for %s!" % self.expr.get_type())

    def __rlshift__(self, other):
        raise NotImplementedError("rlshift is not implemented for %s!" % self.expr.get_type())

    def __rrshift__(self, other):
        raise NotImplementedError("rrshift is not implemented for %s!" % self.expr.get_type())

    def __rand__(self, other):
        return self.__and__(other)

    def __rxor__(self, other):
        raise NotImplementedError("rxor is not implemented for %s!" % self.expr.get_type())

    def __ror__(self, other):
        return self.__or__(other)

def to_pysmt(val):
    '''
    Take a primitive or a Symbolic object and
    return a pysmt expression
    '''
    if isinstance(val, SymbolicObject):
        return val.expr
    elif isinstance(val, int):
        return Int(val)
    elif isinstance(val, bool):
        return TRUE() if val else FALSE()
    elif isinstance(val, str):
        return String(val)
    elif val is None:
        return None
    elif is_instance_userdefined_and_newclass(val):
        # TODO should this be handled somewhere else?
        # user defined class. Decompose it
        attributes = inspect.getmembers(val, lambda a: not(inspect.isroutine(a)))
        attributes = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
        return [to_pysmt(a[1]) for a in attributes]
    else:
        raise NotImplementedError("Wrap doesn't support this type! %s." %type(val))


def is_instance_userdefined_and_newclass(inst):
    '''
    Hack to find out if the object in question was defined by the user
    '''
    cls = inst.__class__
    if hasattr(cls, '__class__'):
        return ('__dict__' in dir(cls) or hasattr(cls, '__slots__'))
    return False