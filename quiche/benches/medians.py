#!/usr/bin/env python3

import argparse
import pathlib
import glob
import json
import pprint
import csv

parser = argparse.ArgumentParser("Exports FEC benchmarks to CSVs suitable for pgfplot")
parser.add_argument("criterion", help="criterion dir, usually in $PROJECT/target/criterion")
parser.add_argument("measurement", help="name of the experiment e.g. decoder_matrix", nargs="+")
args = parser.parse_args()

crit_dir = pathlib.Path(args.criterion).resolve()

data = {}
for measure in args.measurement:
    measuredir = crit_dir / measure
    data[measure] = {}
    with open(f"csvs/{measure}.csv", 'w') as csvfile:
        w = csv.DictWriter(csvfile, fielnames=['x', 'y'])
        w.writeheader()
        for size in glob.iglob("[0-9]*" , root_dir=measuredir):
            estimates = measuredir / size / "new" / "estimates.json"
            with open(estimates) as e:
                json_data = json.load(e)
                estimate = json_data['mean']['point_estimate']
                data[measure][int(size)] = estimate
                w.writerow((size, estimate))
pprint.pprint(data)
