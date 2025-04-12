# Copyright: see copyright.txt

from collections import deque
import logging

from pyexsmt.path_to_constraint import PathToConstraint
from pyexsmt import pred_to_smt
from pyexsmt.symbolic_types import symbolic_object
from pyexsmt.result import Result

from pysmt.shortcuts import *


class ExplorationEngine:
    def __init__(self, funcinv, solver="z3"):
        self.invocation = funcinv
        # the input to the function
        self.symbolic_inputs = {}  # string -> SymbolicObject
        # initialize
        for n in funcinv.get_names():
            self.symbolic_inputs[n] = funcinv.create_arg_value(n)

        self.constraints_to_solve = deque([])
        self.num_processed_constraints = 0

        self.path = PathToConstraint(lambda c : self.add_constraint(c))
        # link up SymbolicObject to PathToConstraint in order to intercept control-flow
        symbolic_object.SymbolicObject.SI = self.path

        self.solver = Solver(solver)
        self.solver.solve() # generate initial values
        # link up SymbolicObject to the Solver to get concrete values during execution
        symbolic_object.SymbolicObject.SOLVER = self.solver 

        # outputs
        self.result = Result(self.path)

    def add_constraint(self, constraint):
        logging.debug("ADDING CONSTRAINT: %s", repr(constraint))
        self.constraints_to_solve.append(constraint)

    def explore(self, max_iterations=0, max_depth=0, funcs=[], mod=None):
        self.path.max_depth = max_depth
        self.path.mod = mod

        self._one_execution(funcs)
        
        iterations = 1
        if max_iterations != 0 and iterations >= max_iterations:
            logging.debug("Maximum number of iterations reached, terminating")
            return self.result

        while not self._is_exploration_complete():
            selected = self.constraints_to_solve.popleft()
            if selected.processed:
                continue		

            logging.debug("SELECTED CONSTRAINT: %s", repr(selected))
            asserts, query = selected.get_asserts_and_query()
            self._find_counterexample(asserts, query)

            if not self.solver.last_result:
                continue

            self._one_execution(funcs, selected)

            iterations += 1			
            self.num_processed_constraints += 1

            if max_iterations != 0 and iterations >= max_iterations:
                logging.debug("Maximum number of iterations reached, terminating")
                break

        return self.result

    # private

    def _is_exploration_complete(self):
        num_constr = len(self.constraints_to_solve)
        if num_constr == 0:
            logging.info("EXPLORATION COMPLETE")
            return True
        else:
            logging.debug("%d constraints yet to solve (total: %d, already solved: %d)", \
            num_constr, self.num_processed_constraints + num_constr, self.num_processed_constraints)
            return False

    def _one_execution(self, funcs=[], expected_path=None):
        logging.debug("EXPECTED PATH: %s", expected_path)

        self.result.record_inputs(self.symbolic_inputs)
        logging.info("USING INPUTS: %s", self.result.generated_inputs[-1])

        self.path.reset(expected_path)

        try:
            ret = self.invocation.call_function(self.symbolic_inputs, funcs)
        except Exception:
            ret = None

        logging.debug("CURRENT CONSTARINT: %s", repr(self.path.current_constraint))
        logging.info("RETURN: %s", ret)

        self.result.record_output(ret)

    def _find_counterexample(self, asserts, query):
        assumptions = [pred_to_smt(p) for p in asserts] + [Not(pred_to_smt(query))]
        logging.debug("SOLVING: %s", assumptions)
        self.solver.solve(assumptions)
