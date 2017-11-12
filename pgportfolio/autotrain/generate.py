from __future__ import print_function, absolute_import, division
import json
import os
import logging
from os import path


def add_packages(config, repeat=1):
    train_dir = "train_package"
    package_dir = path.realpath(__file__).replace('pgportfolio/autotrain/generate.pyc',train_dir)\
        .replace("pgportfolio\\autotrain\\generate.pyc", train_dir)\
                  .replace('pgportfolio/autotrain/generate.py',train_dir)\
        .replace("pgportfolio\\autotrain\\generate.py", train_dir)
    all_subdir = [int(s) for s in os.listdir(package_dir) if os.path.isdir(package_dir+"/"+s)]
    if all_subdir:
        max_dir_num = max(all_subdir)
    else:
        max_dir_num = 0
    indexes = []

    for i in range(repeat):
        max_dir_num += 1
        directory = package_dir+"/"+str(max_dir_num)
        config["random_seed"] = i
        os.makedirs(directory)
        indexes.append(max_dir_num)
        with open(directory + "/" + "net_config.json", 'w') as outfile:
            json.dump(config, outfile, indent=4, sort_keys=True)
    logging.info("create indexes %s" % indexes)
    return indexes

