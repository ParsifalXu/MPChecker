#!/bin/bash

echo "Start at: "$(date)

# chmod +x project
# ./s project

PROJECT=$1

python3 main.py --download $PROJECT
python3 main.py --extract $PROJECT
python3 main.py --find $PROJECT
python3 main.py --match $PROJECT
python3 main.py --filter $PROJECT
python3 main.py --trans $PROJECT
python3 main.py --symex $PROJECT
python3 main.py --put $PROJECT
python3 main.py --solve $PROJECT


# python3 main.py --filter keras
# python3 main.py --filename keras_new_res --put keras
# python3 main.py --trans keras
# python3 main.py --symex keras
# python3 main.py --solve keras > keras_log.log


# python3 main.py --process xxx 


date