import os 
import sys
import time
import argparse

import tools.downloader
import tools.extractor
import tools.finder
import tools.filter
import tools.matcher
import tools.symexecutor
import tools.translator
import tools.processor
import tools.distributor
import tools.solver
import tools.solver



def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", help="Download target project", required=False)
    parser.add_argument("--extract", help="Extract classes and independent functions", required=False)
    parser.add_argument("--find", help="Find parameters and attributes", required=False)
    parser.add_argument("--match", help="check alignment of parameters' featues and quantities", required=False)
    parser.add_argument("--filter", help="Filter out files that have no interdependencies", required=False)
    parser.add_argument("--trans", help="Translate functions", required=False)
    parser.add_argument("--symex", help="Symbolic Execution", required=False)
    parser.add_argument("--process", help="Use OpenAI API to process document", required=False)
    parser.add_argument("--put", help="Put GPT processed result files into the project", required=False)
    parser.add_argument("--solve", help="Solve contraints and find out bad conditions", required=False)
    parser.add_argument("--runall", help="Run All Operations", required=False)
    parser.add_argument("--runallexp", help="Run All Experiments", required=False)


    if len(argv) == 0:
        parser.print_help()
        exit(1)
    opts = parser.parse_args(argv)
    return opts






if __name__ in "__main__":
    opts = parseArgs(sys.argv[1:])
    if opts.download:
        print(f"Downloading the project: {opts.download}")
        tools.downloader.download_library(opts.download)
        print("End of Download")
    
    if opts.extract:
        print(f"Extracting classes and independent functions in the project: {opts.extract}")
        tools.extractor.extract_class_and_independent_function(opts.extract)
        print("Extraction Completed")

    if opts.find:
        print(f"Finding all parameters and attributes in the project: {opts.find}")
        tools.finder.find_params_and_attributes(opts.find)
        print("Find All")

    if opts.match:
        print(f"Checking alignment of parameters' featues and quantities in the project: {opts.match}")
        tools.matcher.match_alignment(opts.match)
        print(f"End of match")
    
    if opts.filter:
        print(f"Filtering out files that have no interdependencies in the project: {opts.filter}")
        tools.filter.filter_params_and_attributes(opts.filter)
        print(f"End of Filter")

    if opts.trans:
        print(f"Translate functions in the project: {opts.trans}")
        t1 = time.time()
        tools.translator.translate_code(opts.trans)
        t2 = time.time()
        print(f"time: {t2-t1}")
        print(f"End of Translation")

    if opts.symex:
        print(f"Implement symbolic execution on functions in the project: {opts.symex}")
        t1 = time.time()
        tools.symexecutor.symex(opts.symex)
        t2 = time.time()
        print(f"time: {t2-t1}")
        print(f"End of Symbolic Execution")

    if opts.process:
        print(f"Processing documentation in the project: {opts.process}")
        tools.processor.process_document(opts.process)
        print(f"End of Process")

    if opts.put:
        print(f"Put GPT processed result files into the project: {opts.put}")
        tools.distributor.distribute_files(opts.put)
        print(f"End of Distribution")
    
    if opts.solve:
        print(f"Solve constraints in the project: {opts.solve}")
        tools.solver.solve_constraints(opts.solve)
        print(f"End of Solve")


    if opts.runall:
        tools.downloader.download_library(opts.runall)
        tools.extractor.extract_class_and_independent_function(opts.runall)
        tools.finder.find_params_and_attributes(opts.runall)
        tools.matcher.match_alignment(opts.runall)
        tools.filter.filter_params_and_attributes(opts.runall)
        tools.translator.translate_code(opts.runall)
        tools.symexecutor.symex(opts.runall)
        tools.solver.solve_constraints(opts.runall)

    if opts.runallexp:
        tools.extractor.extract_class_and_independent_function(opts.runallexp)
        tools.finder.find_params_and_attributes(opts.runallexp)
        tools.matcher.match_alignment(opts.runallexp)
        tools.filter.filter_params_and_attributes(opts.runallexp)
        tools.translator.translate_code(opts.runallexp)
        tools.symexecutor.symex(opts.runallexp)
        tools.distributor.read_from_benchmark(opts.runallexp)
        tools.solver.solve_constraints(opts.runallexp)
        