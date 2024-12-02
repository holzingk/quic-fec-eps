#!/usr/bin/env python3

import argparse
import pathlib
import glob
import json
import pprint
import os
import csv

parser = argparse.ArgumentParser("Exports FEC benchmarks to CSVs suitable for pgfplot")
parser.add_argument("criterion", help="criterion dir, usually in $PROJECT/target/criterion")
parser.add_argument("outputdir", help="Directory to store the csvs")
#parser.add_argument("measurement", help="name of the experiment e.g. decoder_matrix", nargs="+")
args = parser.parse_args()

crit_dir = pathlib.Path(args.criterion).resolve()

measurements = [ measurement for measurement in crit_dir.iterdir() if not os.path.basename(measurement) == "report" ]

data = {}
for measure in measurements:
    measurename = os.path.basename(measure)
    data[measure] = {}
    with open(f"{args.outputdir}/{measurename}.csv", 'w') as csvfile:
        w = csv.DictWriter(csvfile, fieldnames=['x', 'y'])
        w.writeheader()
        for size in glob.iglob("[0-9]*" , root_dir=measure):
            estimates = measure / size / "new" / "estimates.json"
            with open(estimates) as e:
                json_data = json.load(e)
                estimate = json_data['mean']['point_estimate']
                data[measure][int(size)] = estimate
                w.writerow({'x': size, 'y': estimate})
pprint.pprint(data)
