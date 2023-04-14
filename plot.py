from fileinput import filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import sys
from os import listdir
from subprocess import Popen, PIPE
import csv
import copy

cycle = int(sys.argv[1])
data_path = "/mnt/188/b/bfs/dat"
bin_path = "/home/aihcer0119/johnny/bfs/bin"
DATAFILE = sorted(listdir(data_path))
BINFILE = sorted(listdir(bin_path))
data_len = len(DATAFILE)
bin_len = len(BINFILE)
Matrix = [[0 for x in range(data_len)] for y in range(bin_len)] 
f = open("result.csv", mode="w", newline='')
header = copy.deepcopy(BINFILE)
header.insert(0, 'benchmark')
writer = csv.DictWriter(f, fieldnames=header)
for i, bin_file in zip(range(bin_len), BINFILE):
    for j, data_file in zip(range(data_len), DATAFILE):
        print(bin_file, data_file)
        time = 0
        for k in range(cycle):
            process = Popen("{}/{} {}/{} 8".format(bin_path, bin_file, data_path, data_file), shell=True, stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()
            time += float(stdout.split()[-2][:-2].decode('UTF-8'))
        Matrix[i][j] = time / cycle
        
df = pd.DataFrame({BINFILE[i]:Matrix[i] for i in range(bin_len)}, index = DATAFILE)
ax = df.plot.bar(rot=0, figsize=(15,15), width=0.9)
plt.xticks(fontsize=15, rotation=45)
plt.savefig('result.png')

l = [Matrix[i] for i in range(bin_len)]
l.insert(0, DATAFILE)
l = pd.DataFrame(l).T
l.rename(columns = {i:header[i] for i in range(bin_len+1)}, inplace = True)
l.to_csv(f, sep='\t')