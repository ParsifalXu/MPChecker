# Copyright: see copyright.txt

import logging

class Constraint:
    cnt = 0
    """A constraint is a list of predicates leading to some specific
       position in the code."""
    def __init__(self, parent, last_predicate):
        self.inputs = None
        self.predicate = last_predicate
        self.effect = None
        self.processed = False
        self.parent = parent
        self.children = []
        self.id = self.__class__.cnt
        self.__class__.cnt += 1

    def __eq__(self, other):
        """Two Constraints are equal iff they have the same chain of predicates"""
        if isinstance(other, Constraint):
            if not self.predicate == other.predicate:
                return False
            return self.parent is other.parent
        else:
            return False

    def get_asserts_and_query(self):
        self.processed = True
        asserts = self.get_asserts()
        return asserts, self.predicate	       

    def get_asserts(self):
        # collect the assertions
        if self.parent is None:
            return []
        asserts = []
        tmp = self.parent
        while tmp.predicate is not None:
            asserts.append(tmp.predicate)
            tmp = tmp.parent

        return asserts

    def get_length(self):
        if self.parent == None:
            return 0
        return 1 + self.parent.get_length()

    def __str__(self):
        return str(self.predicate) + "  (processed: %s, path_len: %d)" % (self.processed,self.get_length())

    def __repr__(self):
        s = repr(self.predicate) + " (processed: %s)" % (self.processed)
        if self.parent is not None:
            s += "\n  path: %s" % repr(self.parent)
        return s

    def find_child(self, predicate):
        for c in self.children:
            if predicate == c.predicate:
                return c
        return None

    def add_child(self, predicate):
        assert(self.find_child(predicate) is None)
        c = Constraint(self, predicate)
        self.children.append(c)
        return c

