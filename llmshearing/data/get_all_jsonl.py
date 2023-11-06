import os
import sys

root_dir = sys.argv[1]

log = open("jsonl_list.txt", 'w')
for root, ds, fs in os.walk(root_dir):
    for f in fs:
        full_path = os.path.join(root, f)
        relative_path = full_path[len(root_dir):].lstrip(os.sep) 
        print(relative_path)
        log.write(relative_path + "\n") 
