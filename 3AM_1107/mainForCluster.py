import pandas as pd
import numpy as np
import graphicsHCA as graphics
import utilsHCA as utils
import scipy.cluster.hierarchy as hclust
import scipy.spatial.distance as hdist
import sklearn.decomposition as dec
import matplotlib as mpl


fileName = './dataIN/basketball_training_dataset_modified.csv'

# set warning only when opened more than 50 figures
mpl.rcParams['figure.max_open_warning'] = 50
print(mpl.rcParams['figure.max_open_warning'])

# List of options for the runtime,
# keep in the list only the desired options
drawing_options = ['Partition plot in main axes',
                   # 'Plot histograms',
                   'Variable grouping'
                   ]
discriminant_axes = (drawing_options.
                     __contains__('Partition plot in main axes'))
histograms = (drawing_options.
              __contains__('Plot histograms'))
variable_grouping = (drawing_options.
                     __contains__('Variable grouping'))

table = pd.read_csv(fileName, index_col=0, na_values='')
utils.replace_na_df(table)
print(table)
obs = table.index.values
print(obs)
vars = table.columns.values[1:]
print(vars)
X = table[vars].values
Xstd = utils.standardise(X)
# print(Xstd)

# Construct the hierarchy of instances (observations)
methods = list(hclust._LINKAGE_METHODS)
metrics = hdist._METRICS_NAMES
print('Methods: ', methods)
print('Metrics: ', metrics)

method = 'centroid'
distance = metrics[3]  # citiblock

if (method == 'ward' or method == 'centroid' or
        method == 'median' or method == 'weighted'):
    distance = 'euclidean'
else:
    distance = 'cityblock'

h = hclust.linkage(Xstd, method=method, metric=distance)
print('Matrix of junctions and distances between nodes:\n', h)

m = np.shape(h)[0]  # Maximum number of junctions

k = m - np.argmax(h[1:m, 2] - h[:(m - 1), 2])
g_max, codes = utils.cluster_distribution(h, k)
# cluster display
utils.cluster_display(g_max, table.index, 'Code',
                      './dataOUTPUT/ClusterDistribution.csv')
# save cluster distribution in CSV file
utils.cluster_save(g=g_max, row_labels=table.index.values,
                       col_label='Cluster',
                   file_name='ClusterGrouping.csv')

t_1, j_1, m_1 = utils.threshold(h)
print('threshold=', t_1, 'junction with the maximum difference=', j_1,
      'no. of junctions=', m_1)

# Determination of cluster colors
color_clusters = utils.color_clusters(h, k, codes)
graphics.dendrogram(h, labels=obs,
            title='Observations clustering. Partition of maximum stability. '
                  'Method: ' +
            method + ' Metric: ' + distance,
                    colors=color_clusters, threshold=t_1)

# Variable hierarchy
if variable_grouping:
    # method_v = methods[5]  # average
    method_v = 'average'
    distance_v = 'correlation'

    h2 = hclust.linkage(X.transpose(), method=method_v, metric=distance_v)
    t_2, j_2, m_2 = utils.threshold(h2)
    print('threshold=', t_2, 'junction with the maximum difference=', j_2,
          'no. of junctions=', m_2)
    graphics.dendrogram(h2, labels=vars,
                    title="Variable clustering. Method: " + method_v +
                    " Metric: " + distance_v, threshold=t_2)

n = np.shape(X)[0]
# Prepare a selection list for the partitions,
# starting from the partition with two clusters
list_selections = [str(i) + ' clusters' for i in range(2, n - 1)]
partitions = list_selections[0:5]

# Create DataFrame with the maximum stability partition
# and the selected partitions
# TODO

graphics.show()

