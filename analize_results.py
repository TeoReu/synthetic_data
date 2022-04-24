import pandas as pd

f = pd.read_csv('old_results.txt', index_col=None)

dgi_2_graphs = f[f['Model'] =='dgi_hetero_graph']

dgi_2_graphs.to_csv('dgi_hetero_graph.csv')