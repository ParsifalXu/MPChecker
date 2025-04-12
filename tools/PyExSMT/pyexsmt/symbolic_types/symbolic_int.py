# Copyright: copyright.txt

from pyexsmt.symbolic_types.symbolic_object import SymbolicObject, to_pysmt

from pysmt.shortcuts import *

class SymbolicInteger(SymbolicObject):
    def __init__(self, expr, name = "se"):
        SymbolicObject.__init__(self, expr, name, INT)

    ## LOGICAL OPERATORS
    def __and__(self, other):
        raise NotImplementedError("and is not implemented for %s!" % self.expr.get_type())

    def __or__(self, other):
        raise NotImplementedError("or is not implemented for %s!" % self.expr.get_type())

    ## ARITHMETIC OPERATORS
    def __add__(self, other):
        other = to_pysmt(other)
        if self.expr.get_type() != other.get_type():
            raise TypeError("CANNOT '+' %s and %s" %(self.expr.get_type(), other.get_type()))
        return SymbolicInteger(self.expr + other)

    def __sub__(self, other):
        other = to_pysmt(other)
        if self.expr.get_type() != other.get_type():
            raise TypeError("CANNOT '-' %s and %s" %(self.expr.get_type(), other.get_type()))
        return SymbolicInteger(self.expr - other)

    def __mul__(self, other):
        other = to_pysmt(other)
        if self.expr.get_type() != other.get_type():
            raise TypeError("CANNOT '*' %s and %s" %(self.expr.get_type(), other.get_type()))
        return SymbolicInteger(self.expr * other)

    def __mod__(self, other):
        other = to_pysmt(other)
        if self.expr.get_type() != other.get_type() or self.expr.get_type() != INT:
            raise TypeError("CANNOT 'mod' %s and %s" %(self.expr.get_type(), other.get_type()))
        return SymbolicInteger(self.expr % other)


    def __floordiv__(self, other):
        other = to_pysmt(other)
        if self.expr.get_type() != other.get_type() or self.expr.get_type() != INT:
            raise TypeError("CANNOT '//' %s and %s" %(self.expr.get_type(), other.get_type()))
        return SymbolicInteger(self.expr // other)

    ## UNARY OPERATORS
    def __neg__ (self):
        return SymbolicInteger(Int(0) - self.expr)

    def __pos__ (self):
        return self

    def __abs__ (self):
        return SymbolicInteger(Ite(self.expr < 0, -self.expr, self.expr))

    ## REVERSE OPERATORS
    def __radd__ (self, other):
        return self.__add__(other)
    
    def __rmul__ (self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        return (-self).__add__(other)