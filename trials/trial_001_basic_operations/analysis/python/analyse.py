#!/usr/bin/env python3
"""
python3 analyze.py --results "results/*.json" --targets target_values.json
"""

import argparse
import glob
import json
import math
import os
import numpy
import statistics
from collections import defaultdict
from typing import List, Dict, Any
from typing import Any, Dict, Iterable, List, Tuple
import matplotlib.pyplot as plt




class Analyze:

    operations_list = []
    precise = {}
    results = { 
                "results": [],
                "sorted": {}, 
                "collated": {}, 
                "statistics": {} 
              }
    output_json = ""
    output_dir = ""

    @staticmethod
    def load_file(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if("expected_value" in data.keys()):
                data["expected_value"] = float(data["expected_value"])
            if("calculated_value" in data.keys()):
                data["calculated_value"] = float(data["calculated_value"])
            if("values" in data.keys()):
                data["values"] = numpy.array(data["values"], dtype=numpy.float64)

            return data
        except:
            return None


    @staticmethod
    def load_precise(directory):
        pattern = os.path.join(directory, "precise*.json")
        for file_path in glob.glob(pattern):
            if not os.path.isfile(file_path):
                continue
            print(file_path)
            file_data = Analyze.load_file(file_path)
            print(file_data)
            if(file_data is not None):
                Analyze.precise[file_path] = file_data


    @staticmethod
    def process_files(directory):
        pattern = os.path.join(directory, "*.json")
        for file_path in glob.glob(pattern):
            if not "precise" in file_path:
                if not os.path.isfile(file_path):
                    continue
                file_data = Analyze.load_file(file_path)
                if(file_data is not None):

                    # Results
                    operation = file_data["operation"]
                    calculated_value = file_data["calculated_value"]
                    method = file_data["method"]
                    device = file_data["device"]
                    id = method + " " + device + ": " + operation
                    
                    # Precise values
                    precise = Analyze.get_precise(operation)
                    expected_value = precise["expected_value"]

                    result = {
                        "file": file_path,
                        "operation": operation,
                        "expected_value": expected_value,
                        "calculated_value": calculated_value,
                        "difference": (calculated_value - expected_value),
                        "iterations": file_data["iterations"],
                        "time_per_iteration": file_data["time_per_iteration"],
                        "method": method,
                        "device": device,
                        "id": id
                    }
                    Analyze.results["results"].append(result)


    @staticmethod
    def get_precise(operation):
        for key in Analyze.precise.keys():
            if(operation == Analyze.precise[key]["operation"]):
                return Analyze.precise[key]
            

    @staticmethod
    def save_data(args):       
        with open(Analyze.output_json, "w") as f:
            json.dump(Analyze.results, f, indent=4)


    @staticmethod
    def load_data():  
        Analyze.results = Analyze.load_file(Analyze.output_json)
            
            

    @staticmethod
    def sort_results():       
      
        for i in range(len(Analyze.results["results"])):
            result = Analyze.results["results"][i]
            operation = result["operation"]  
            print(operation)

            if(operation not in Analyze.operations_list):
                Analyze.operations_list.append(operation)
        print(Analyze.operations_list)


    @staticmethod
    def plot_time_per_iteration(): 

        for operation in Analyze.results["statistics"].keys():
            plot_path = os.path.join(Analyze.output_dir, operation + ".png")

            x_labels = []
            y_values = []

            for id in Analyze.results["statistics"][operation]["time_per_iteration"].keys():
                x_labels.append(id)
                y_values.append(Analyze.results["statistics"][operation]["time_per_iteration"][id]["min_value"])


            plt.figure()
            plt.bar(range(len(x_labels)), y_values)
            plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
            #plt.ylabel(stat)
            #plt.title(f"{op} — {stat}")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=200)
            plt.close()

    



    @staticmethod
    def main():

        # Arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("--results", default="results", help="glob for result JSON files")
        ap.add_argument("--outdir", default="analysis", help="output JSON summary file")
        ap.add_argument("--outjson", default="summary.json", help="output JSON summary file")
        args = ap.parse_args()

        Analyze.output_dir = args.outdir
        Analyze.output_json = os.path.join(args.outdir, args.outjson)
        os.makedirs(args.outdir, exist_ok=True)
        print("Output json: ", Analyze.output_json)

        """
        Analyze.load_precise(args.results)
        Analyze.get_precise("add")
        Analyze.process_files(args.results)
        Analyze.save_data()
        """
        
        Analyze.load_data()
        Analyze.sort_results()
        #Analyze.statistics()
        #Analyze.plot_time_per_iteration()



if(__name__ == "__main__"):
    raise SystemExit(Analyze.main()) 











"""



    @staticmethod
    def statistics():       
        print("Statistics")
        Analyze.results = Analyze.load_file(Analyze.output_json)

        Analyze.results["collated"] = {}
        for result in Analyze.results["results"]:
            operation = result["operation"]
            if(operation not in Analyze.results["collated"].keys()):
                Analyze.results["collated"][operation] = {}
            Analyze.results["collated"][operation]["time_per_iteration"] = {}

        for result in Analyze.results["results"]:
            operation = result["operation"]
            id = result["method"] + " " + result["device"] + ": " + result["operation"]
            if(id not in Analyze.results["collated"][operation]["time_per_iteration"].keys()):
                Analyze.results["collated"][operation]["time_per_iteration"][id] = []


        for result in Analyze.results["results"]:
            operation = result["operation"]
            id = result["method"] + " " + result["device"] + ": " + result["operation"]
            
            Analyze.results["collated"][operation]["time_per_iteration"][id].append(result["time_per_iteration"])    
            
        print(Analyze.results["collated"])


        #Analyze.statistics

        for operation in Analyze.results["collated"].keys():
            if(operation not in Analyze.results["statistics"].keys()):
                Analyze.results["statistics"][operation] = {"time_per_iteration": {}}

            for id in Analyze.results["collated"][operation]["time_per_iteration"].keys():
                Analyze.results["statistics"][operation]["time_per_iteration"][id] = {}

                Analyze.results["statistics"][operation]["time_per_iteration"][id]["min_value"] = min(Analyze.results["collated"][operation]["time_per_iteration"][id])
                Analyze.results["statistics"][operation]["time_per_iteration"][id]["max_value"] = max(Analyze.results["collated"][operation]["time_per_iteration"][id])



def load_targets(directory):
    target_values_json = os.path.join(directory, "target_values.json")
    targets = {}

    for op in operations_list:
        targets[op] = {}
        target_values_json = os.path.join(directory, "precise_values_" + op + ".json")
        with open(target_values_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        targets[op]["expected_value"] = float(data["expected_value"])
        try:
            targets[op]["values"] = [float(v) for v in data.get("values", [])]
        except (TypeError, ValueError):
            continue


    return targets



def load_results(directory):
    results = []
    pattern = os.path.join(directory, "*.json")
    for filepath in glob.glob(pattern):
        if not os.path.isfile(filepath):
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

                try:
                    calculated_value = float(data.get("calculated_value"))
                    data["calculated_value"] = calculated_value
                except (TypeError, ValueError):
                    continue
                try:
                    values = [float(v) for v in data.get("values", [])]
                    data["values"] = values
                except (TypeError, ValueError):
                    continue

                results.append(data)
        except Exception as e:
            print(f"Skipping {filepath}: {e}")

    return results



def process(
    results: List[Dict[str, Any]],
    target_values: Dict[str, float],
) -> Dict[str, Dict[str, Dict[str, float]]]:

    grouped_iters = defaultdict(lambda: defaultdict(list))
    grouped_pct_err = defaultdict(lambda: defaultdict(list))

    # --- group ---
    for r in results:
        operation = r.get("operation")
        method = r.get("method")
        iterations = r.get("iterations")
        calculated_value = r.get("calculated_value")

        if not isinstance(operation, str):
            continue
        if not isinstance(method, str):
            continue
        if not isinstance(iterations, int):
            continue
        if not isinstance(calculated_value, (int, float)):
            continue
        if operation not in target_values:
            continue

        expected_value = target_values[operation]["expected_value"]

        # avoid divide-by-zero
        if expected_value == 0:
            continue

        pct_error = ((calculated_value - expected_value) / expected_value) * 100.0

        grouped_iters[operation][method].append(iterations)
        grouped_pct_err[operation][method].append(pct_error)

    # --- compute stats ---
    processed = {}

    for operation, methods in grouped_iters.items():
        processed[operation] = {}

        for method, iters in methods.items():
            pct_errors = grouped_pct_err[operation][method]

            processed[operation][method] = {
                "min_iters": min(iters),
                "max_iters": max(iters),
                "mean_iters": statistics.fmean(iters),
                "median_iters": statistics.median(iters),
                "mean_percentage_error": statistics.fmean(pct_errors),
                "max_abs_percentage_error": max(abs(e) for e in pct_errors),
            }

    return processed



def save_json(processed, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2)




def plot_iterations(
    processed: Dict[str, Dict[str, Dict[str, float]]],
    stat: str = "mean_iters",
    operations: Iterable[str] | None = None,
) -> None:

    if operations is None:
        ops = sorted(processed.keys())
    else:
        ops = [op for op in operations if op in processed]

    for op in operations_list:
        methods = processed.get(op, {})
        if not methods:
            continue

        x_labels = sorted(methods.keys())
        y = [methods[m].get(stat) for m in x_labels]

        # drop missing
        pairs = [(m, v) for m, v in zip(x_labels, y) if isinstance(v, (int, float))]
        if not pairs:
            continue

        x_labels, y = zip(*pairs)

        plt.figure()
        plt.bar(range(len(x_labels)), y)
        plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
        plt.ylabel(stat)
        plt.title(f"{op} — {stat}")
        plt.tight_layout()
        plt.savefig(os.path.join("analysis", op + ".png"), dpi=200)
        plt.close()



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="glob for result JSON files")
    ap.add_argument("--targets", default="target_values", help="path to target_values")
    ap.add_argument("--outdir", default="analysis", help="output JSON summary file")
    ap.add_argument("--outjson", default="summary.json", help="output JSON summary file")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    target_values = load_targets(args.targets)
    results = load_results(args.results)
    processed = process(results, target_values)

    out_path = os.path.join(args.outdir, args.outjson)
    save_json(processed, out_path)

    print(processed)

    plot_iterations(processed, stat="median_iters", operations=["add"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main()) 

"""

