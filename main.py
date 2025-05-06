import os 
import sys
import time
import json
import shutil
import argparse
import pandas as pd
import subprocess as sub

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
    parser.add_argument("--runoneexp", help="Run One Experiments", required=False)
    parser.add_argument("--runallexp", help="Run All Experiments", required=False, action="store_true")
    parser.add_argument("--test", help="Run All Experiments", required=False, action="store_true")


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


    def clear_folder(folder_path):
        if not os.path.exists(folder_path):
            print(f"Not exist: {folder_path}")
            return

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Fail to delete {file_path}, the reason is {e}")

    if opts.runoneexp:
        clear_folder('./info')
        tools.extractor.extract_class_and_independent_function(opts.runoneexp)
        tools.finder.find_params_and_attributes(opts.runoneexp)
        tools.matcher.match_alignment(opts.runoneexp)
        tools.filter.filter_params_and_attributes(opts.runoneexp)
        tools.translator.translate_code(opts.runoneexp)
        tools.symexecutor.symex(opts.runoneexp)
        tools.distributor.read_from_benchmark(opts.runoneexp)
        tools.solver.solve_constraints(opts.runoneexp)
    
    if opts.runallexp:
        t1 = time.time()
        if not os.path.exists("./logs"):
            os.makedirs("./logs")
        for i in range(1, 73):
            for j in range(0, 3):
                print(f"===>>> Running experiment c{i}-{j} ===>>>")
                command = f"python3 main.py --runoneexp c{i}-{j}"
                print(command, "\n")
                sub.run(command, shell=True, stdout=open(f"./logs/c{i}-{j}.log", 'w'), stderr=sub.STDOUT, timeout=30)
           
        inconsistent = 0
        consistent = 0
        fp = 0
        fn = 0

        result = {}
        for filename in os.listdir("./logs"):
            if filename.endswith(".log"):
                filepath = os.path.join("./logs", filename)
                with open(filepath, 'rb') as f:
                    try:
                        f.seek(-2, os.SEEK_END)
                        while f.read(1) != b'\n':
                            f.seek(-2, os.SEEK_CUR)
                    except OSError:
                        f.seek(0)
                    last_line = f.readline().decode().strip()
                
                prefix, rest = filename.split('-', 1)
                id = os.path.splitext(rest)[0]
                cx = prefix
    
                status = not last_line.startswith("BAD CONSTRAINT")

                with open(f"./benchmark.json", "r") as f:
                    data = json.load(f)
                oracle_value = next((item['oracle'] for item in data if item['id'] == f"{cx}-{id}"), None)
                    
                if status != (oracle_value.lower() == "true"):
                    if status:
                        status = "FN"
                        fn += 1
                    else:
                        status = "FP"
                        fp += 1
                else:
                    if status:
                        consistent += 1
                    else:
                        inconsistent += 1


                if cx not in result:
                    result[cx] = {}
                result[cx][int(id)] = status
        sorted_keys = sorted(result.keys(), key=lambda x: (x[0], int(x[1:])))
        df = pd.DataFrame.from_dict({k: result[k] for k in sorted_keys}, orient='index').sort_index(axis=1)
        df.index.name = "id"

        df.to_excel("result.xlsx")

        print("===>>> Result: ===>>>")
        print(f"Total: {consistent + inconsistent + fp + fn}")
        print(f"Consistent: {consistent}")
        print(f"Inconsistent: {inconsistent}")
        print(f"False Positive: {fp}")
        print(f"False Negative: {fn}")

        t2 = time.time()
        print(f"time: {t2-t1}")
        print(f"End of Symbolic Execution")