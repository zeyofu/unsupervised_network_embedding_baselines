import numpy as np
import pickle as pkl
import scipy.sparse as sp
from tqdm import tqdm


""" should put this file inside data dir """

feature = 'node-feature.csv'
link = 'link.csv'
diffusion = 'author-1900-2020.csv'
diff_content = 'paper-feature.csv'
eval = 'eval/rel.txt'

""" convert features to np.array """
feature_file = open(feature, 'r').readlines()
feature_len = len(feature_file[0].rstrip().split(',')[1].split())

num_node = 0
for feature in feature_file:
    nodeid = int(feature.rstrip().split(',')[0])
    if nodeid > num_node:
        num_node = nodeid
num_node += 1 #pad one because the nodeid start from 1 .....

print('number of node', num_node, 'feature dimension', feature_len)
embedding = np.zeros((num_node, feature_len))
for feature in tqdm(feature_file):
    nodeid, node_feature = feature.rstrip().split(',')
    embedding[int(nodeid)] = [float(i) for i in node_feature.split()]

with open('features.p','wb') as feature_file:
    pkl.dump(embedding, feature_file, pkl.HIGHEST_PROTOCOL)

""" convert links to sp_matrix and graph """
dim = num_node
row_id_list = []
col_id_list = []
data_list = []
max_row_id = 0
max_col_id = 0
graph = {}

link_file = open(link, 'r').readlines()[1:]
links = [tuple(map(int, i.rstrip().split(','))) for i in link_file]
link2weight = {}
for pair in links:
    if pair not in link2weight:
        link2weight[pair] = 1
    else:
        link2weight[pair] += 1

for pair in tqdm(links):
    row_id_list.append(pair[0])
    max_row_id = max(max_row_id, pair[0])
    col_id_list.append(pair[1])
    max_col_id = max(max_col_id, pair[1])
    data_list.append(link2weight[pair])
    if pair[0] not in graph: graph[pair[0]] = []
    if pair[1] not in graph: graph[pair[1]] = []
    graph[pair[0]].append(pair[1])
    graph[pair[1]].append(pair[0])
print(max_row_id, max_col_id, dim)
dim = max(max_row_id, max_col_id) + 1
matrix_dim = dim
sparse_affnity = sp.csr_matrix((data_list, (row_id_list, col_id_list)), shape=(dim, dim))
for i in range(matrix_dim):
    if i in graph:
        graph[i].sort()

with open('affinity_matrix.p','wb') as affinity_output:
    pkl.dump(sparse_affnity, affinity_output, pkl.HIGHEST_PROTOCOL)
with open('graph.p','wb') as graph_o:
    pkl.dump(graph, graph_o, pkl.HIGHEST_PROTOCOL)


""" construct diffusion data """
diff_graph = {}
diff_file = open(diffusion, 'r').readlines()[1:]
for line in diff_file:
    author, paper = map(int, line.rstrip().split(','))
    if paper not in diff_graph:
        diff_graph[paper] = [author]
    else:
        diff_graph[paper].append(author)

count = 0
for paper, authors in diff_graph.items():
    if len(authors) >= 20:
        count += 1
print(count)

with open('diff-sub.csv', 'w') as of:
    of.write('paperid,author_list\n')
    for paper, authors in diff_graph.items():
        if len(authors) > 20:
            count+=1
        of.write(','.join(map(str, [paper]+authors)))
        of.write('\n')

diff_content_file = open(diff_content, 'r').readlines()
diff_feature = {}
for line in tqdm(diff_content_file):
    paper, feature = line.rstrip().split(',')
    paper = int(paper)
    feature = np.array(list(map(float, feature.split())))
    diff_feature[paper] = feature

diff = {}
for paper, authors in diff_graph.items():
    if len(authors) < 20:
        continue
    diff[paper] = (np.array(authors), diff_feature[paper])

with open('diffusion.p', 'wb') as diff_o:
    pkl.dump(diff, diff_o, pkl.HIGHEST_PROTOCOL)

""" construct small graph """
# labled_nodes = set()
# with open(eval, 'r') as inf:
#     for l in inf:
#         n1, n2, _ = l.rstrip().split('\t')
#         labled_nodes.update([n1, n2])
#
# graph = {}
# link_file = open(link, 'r').readlines()[1:]
# links = [tuple(i.rstrip().split(',')) for i in link_file]
# link2weight = {}
# for pair in tqdm(links):
#     if pair[0] not in graph: graph[pair[0]] = []
#     if pair[1] not in graph: graph[pair[1]] = []
#     graph[pair[0]].append(pair[1])
#     graph[pair[1]].append(pair[0])
#
# all_nodes = labled_nodes.copy()
# for node in labled_nodes:
#     all_nodes.update(graph[node])
# all_nodes2 = all_nodes.copy()
# for node in all_nodes:
#     all_nodes2.update(graph[node])
#
# with open('link-sub.csv', 'w') as of:
#     of.write('author_id,author_id\n')
#     for n1, n2 in tqdm(links):
#         if n1 in all_nodes2 and n2 in all_nodes2:
#             of.write(f'{n1},{n2}\n')
# with open('node-feature-sub.csv', 'w') as of, open(feature, 'r') as inf:
#     for l in tqdm(inf):
#         nodeid = l.rstrip().split(',')[0]
#         if nodeid in all_nodes2:
#             of.write(l)
# with open('author-1900-2020-sub.csv', 'w') as of, open(diffusion, 'r') as inf:
#     next(inf)
#     of.write('author_id,paper_id\n')
#     for line in tqdm(inf):
#         author, _ = line.rstrip().split(',')
#         if author in all_nodes2:
#             of.write(line)
