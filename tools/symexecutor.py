import os 
import subprocess as sub

from tools.macros import INFO_DIR, PYEXECUTOR


def symex(project):
    for root, dirs, files in os.walk(f"{INFO_DIR}/{project}"):
        for file in files:
            if file.endswith('_trans.py'):
                single_symex(root, file)

def single_symex(root, file):
    entryfunc = file[:-9]
    log_path = f"{root}/{entryfunc}_path.txt"
    print(f"path: {log_path}")
    command = f"python3 {PYEXECUTOR} --path {root}/{file} --entry {entryfunc}"
    print(command)
    try:
        sub.run(command, shell=True, stdout=open(log_path, 'w'), stderr=sub.STDOUT, timeout=30)
    except:
        print("Timeout")

    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            pass
        f.close()

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as fr:
        first_line = fr.readline().strip()
    if first_line.startswith('Traceback'):
        with open(log_path, 'w') as fw:
            pass
        fw.close()