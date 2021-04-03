import os
import csv

def write_results(experiment_id, results):
    header = ["Name", "Build Time (s)", "Evaluation Time (s)", 
              "Evaluation Error (MSE)","Memory Size (KB)"]
    with open(os.path.join("results","{}.csv".format(experiment_id)), 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        for each in results:
            csvwriter.writerow(each)