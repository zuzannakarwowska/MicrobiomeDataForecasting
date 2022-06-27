import pandas as pd
import sys


file = sys.argv[1]

df = pd.read_csv(file, sep = '\t', index_col = [0])
df.to_csv('table.tsv', sep = '\t')
