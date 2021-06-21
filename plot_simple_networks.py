#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:00:11 2021

@author: klaudiamur
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import collections

#### create and plot simple networks from adjacency matrix


def create_random_network(n, directed, weighted):
    if weighted == False:
        
        matrix = np.random.randint(2, size=(n, n))
        for i in range(n):
            matrix[i][i] = 0
            
        if directed == False:
            for i in range(n):
                for j in range(i):
                    matrix[i][j] = matrix[j][i]
                    
                    
    else:
        #matrix = 
        matrix = np.random.randint(1, 11, size=(n,n)) * np.random.randint(2, size=(n, n))*0.1
        for i in range(n):
            matrix[i][i] = 0
            
        if directed == False:
            for i in range(n):
                for j in range(i):
                    matrix[i][j] = matrix[j][i]
         
    if directed == True:
        G = nx.DiGraph(matrix)
    else:
        G = nx.Graph(matrix)
                            
    return matrix, G          
    

def create_bridge(n):
    matrix = np.zeros((n, n), dtype = int)
    #b = np.random.randint(n)
    b = int(np.floor(n/2))
    for i in range(b):
        for j in range(i):
            matrix[i][j] = np.random.choice(2, p = [0.2, 0.8])
            
    for i in range(b-1, n):
        for j in range(b-1, i):
            matrix[i][j] = np.random.choice(2, p = [0.2, 0.8])
            
    for j in range(n):
        matrix[i][i] = 0
        for i in range(j):
            matrix[i][j] = matrix[j][i]
            
    G = nx.Graph(matrix)
    
    return matrix, G, b


def create_latex_matrix(matrix):
    matrix_latex = '\begin{pmatrix}'
    for i in range(len(matrix)):
        for j in matrix[i][:-1]:
            matrix_latex = matrix_latex + str(j) + ' & '
        j =  matrix[i][-1]
        matrix_latex = matrix_latex + str(j)
        matrix_latex = matrix_latex + ' \\ '
    matrix_latex = matrix_latex + ' \end{pmatrix}'
    return matrix_latex


def draw_network(G):
    #widths = nx.get_edge_attributes(G, 'weight')
    
    plt.figure(frameon=False, dpi=500, figsize=(6,4))
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog = 'neato')
    nx.draw_networkx_nodes(G, pos, node_color='Brown', node_size = 5)
   # nx.draw_networkx_edges(G, pos, edgelist = widths.keys(),
    #                       width=list(widths.values()))
    nx.draw_networkx_edges(G, pos, width = 0.5)
    #nx.draw_networkx_labels(G, pos)
    plt.show()
    


# =============================================================================
# Plot clustering coefficient and stuff like that
# =============================================================================
## plot a triple

edges_triple = [(0, 1), (1, 2), (2, 0)]

G = nx.Graph(edges_triple)
node_labels = {0:'A', 1:'B', 2:'C'}

pos = nx.drawing.nx_agraph.graphviz_layout(G, prog = 'dot')
fig = plt.figure(frameon=False, dpi=500, figsize=(5,5))

nx.draw_networkx_nodes(G, pos, 
                       #node_color=list(ec.values()), 
                       node_color = 'Purple',
                       node_size= 1000,
                       )

nx.draw_networkx_edges(G, pos, width = 1, alpha = 0.8, 
                       edgelist=[(2, 0), (1, 2)]
                       )

#nx.draw_networkx_edges(G, pos, width = 0.6, alpha = 1, edgelist=[(3,6),  (6,10), (14, 10), (14, 3)])
nx.draw_networkx_labels(G, pos,  font_color='white', labels=node_labels)
plt.show()
fig.savefig('/Users/klaudiamur/Box/Thesis/coding/graphics/triangle3.png', transparent=True, frameon=False)




# =============================================================================
# plt ego networks yeeey!!
# =============================================================================
n = 11

edgelist = [(0, i) for i in range(1, n)]
#edges_pick = np
edges1 = np.random.choice(n, size= 30)
edges2 = np.random.choice(n, size = 30)

el_tot = [(i, j) for i in range(n) for j in range(i+1, n)]

edgelist2 = [(i, j) for i, j in zip(edges1, edges2)]

el = edgelist + edgelist2

G = nx.Graph(el_tot)
#G = nx.Graph(edgelist)
labels = {0:'i'}

#pos = nx.drawing.nx_agraph.graphviz_layout(G, prog = 'neato')
#pos = {k:(v[0])}
fig = plt.figure(frameon=False, dpi=500, figsize=(5,5))

nx.draw_networkx_nodes(G, pos, 
                       #node_color=list(ec.values()), 
                       node_color = 'Purple',
                       node_size= 200,
                       )

nx.draw_networkx_edges(G, pos, width = 1, alpha = 0.3,
                       #edgelist=[(0, 1), (0, 2)]
                       )

nx.draw_networkx_edges(G, pos, width = 1, alpha = 0.8, edgelist = edgelist,
                       #edgelist=[(0, 1), (0, 2)]
                       )


#nx.draw_networkx_edges(G, pos, width = 0.6, alpha = 1, edgelist=[(3,6),  (6,10), (14, 10), (14, 3)])
#nx.draw_networkx_labels(G, pos,  font_color='white', labels=node_labels)
nx.draw_networkx_labels(G, pos,  font_color='white', labels=labels, font_weight=1)
plt.show()

fig.savefig('/Users/klaudiamur/Box/Thesis/coding/graphics/1clustering.png', transparent=True, frameon=False)




# =============================================================================
# Plot degree distributions!! (scale-free + random network)
# second: plot on logarithmic scale!
# =============================================================================
n = 40
m = 200
m2 = int(np.floor(m/n))

G_rand = nx.gnm_random_graph(n, m)
G_sf = nx.barabasi_albert_graph(n, m2)

G = G_rand
pos = nx.drawing.nx_agraph.graphviz_layout(G, prog = 'neato')
fig = plt.figure(frameon=False, dpi=500, figsize=(5,5))

nx.draw_networkx_nodes(G, pos, 
                       #node_color=list(ec.values()), 
                       node_color = 'Purple',
                       node_size= 100,
                       alpha = 1,
                       )

nx.draw_networkx_edges(G, pos, width = 1, alpha = 0.3,
                       #edgelist=[(0, 1), (0, 2)]
                       )

#nx.draw_networkx_edges(G, pos, width = 1, alpha = 0.8, edgelist = edgelist,
                       #edgelist=[(0, 1), (0, 2)]
 #                      )


#nx.draw_networkx_edges(G, pos, width = 0.6, alpha = 1, edgelist=[(3,6),  (6,10), (14, 10), (14, 3)])
#nx.draw_networkx_labels(G, pos,  font_color='white', labels=node_labels)
#nx.draw_networkx_labels(G, pos,  font_color='white', labels=labels, font_weight=1)
plt.show()




degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color="b")
plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xticks([d + 0.4 for d in deg])
ax.set_xticklabels(deg)

plt.show()



def plot_degree_histogram(g, normalized=True):
    print("Creating histogram...")
    aux_y = nx.degree_histogram(g)
    
    aux_x = np.arange(0,len(aux_y)).tolist()
    
    n_nodes = g.number_of_nodes()
    
    if normalized:
        for i in range(len(aux_y)):
            aux_y[i] = aux_y[i]/n_nodes
    
    return aux_x, aux_y

aux_x1, aux_y1 = plot_degree_histogram(G_rand)
aux_x2, aux_y2 = plot_degree_histogram(G_sf)

plt.title('\nDistribution Of Node Linkages (log-log scale)')
plt.xlabel('Degree\n(log scale)')
plt.ylabel('Number of Nodes\n(log scale)')
plt.xscale("log")
plt.yscale("log")
plt.plot(aux_x1, aux_y1, 'o', c = 'Purple')
plt.plot(aux_x2, aux_y2, 'o', c= 'r')


# =============================================================================
# Make a network of m subnetworks (cliques?) with connections between them! 
# =============================================================================
m = 15
n_min = 5
n_max = 40
n_between = 4

rng = np.random.default_rng()
a = 0.5

s = rng.power(a, m)
sizes_subnetworks = [int(np.floor(i * (n_max - n_min) + n_min )) for i in s]


G0 = nx.barabasi_albert_graph(sizes_subnetworks[0], n_min-1)
for i in range(1, m):
    size = sizes_subnetworks[i]
    size_G = len(G0.nodes)
    G1 = nx.barabasi_albert_graph(size, n_min-1)
    G1 = nx.relabel_nodes(G1, lambda x: x+size_G)
    
    G0 = nx.compose(G0, G1)
    
    ### create random connections to the rest of the graph:
    
    nodes_old= np.random.choice(size_G, size=n_between, replace=False)
    nodes_new = np.random.choice(np.arange(size_G, size_G + size), size=n_between, replace=False)
    
    for node_1, node_2 in zip(nodes_old, nodes_new):
        G0.add_edge(node_1, node_2)
        
        

node_colors = [[1/m * i  for _ in range(sizes_subnetworks[i])] for i in range(m)]
node_colors = [item for sublist in node_colors for item in sublist]

node_size = [np.random.random() * 100 + 50 for node in node_colors] ## mit leichtem noise?
alpha = [0.9 - np.random.random() * 0.3 for node in node_colors]

pos = nx.drawing.nx_agraph.graphviz_layout(G0, prog = 'neato')
fig = plt.figure(frameon=False, dpi=1200, figsize=(6,4))

nx.draw_networkx_nodes(G0, pos, 
                       #node_color=list(ec.values()), 
                       node_color = node_colors,
                       node_size= node_size,
                       vmin = 0,
                       vmax = 1,
                       cmap='twilight', 
                       #node_size = 50,
                       alpha = alpha,
                       #alpha = 0.85,
                       linewidths = 0
                       )
# nx.draw_networkx_edges(G, pos, edgelist = widths.keys(),
 #                       width=list(widths.values()))
nx.draw_networkx_edges(G0, pos, width = 0.8, alpha = 0.2, edgelist=None)
#nx.draw_networkx_edges(G, pos, width = 0.6, alpha = 1, edgelist=[(3,6),  (6,10), (14, 10), (14, 3)])
#nx.draw_networkx_labels(G, pos,  font_color='white')
plt.show()

fig.savefig('/Users/klaudiamur/Google Drive/Commwork/Pitch, Introduction documents/graphs/Pitch_presentation/overall_nw3.png', transparent=True)






       
#draw_network(G0)    
    






    
matrix, G, b = create_bridge(10)    



G = nx.barabasi_albert_graph(30, 2)
G = nx.powerlaw_cluster_graph(20, 3, 0.8)
draw_network(G)












    
n = 6
#matrix, G = create_random_network(n, directed = False, weighted = False)
#matrix, G, b = create_bridge(n)
#draw_network(G)

#G = nx.gnm_random_graph(n, n*2)

## make a network with a node that has high eigenvector centrality but low degree centrality


G1 = nx.barabasi_albert_graph(n, 3)
G2 = nx.barabasi_albert_graph(n, 2)
G3 = nx.barabasi_albert_graph(n, 3)

G2 = nx.relabel_nodes(G2, lambda x: x+n)
G3 = nx.relabel_nodes(G3, lambda x: x+2*n)

dg1 = {n:d for n, d in G1.degree()}
dg2 = {n:d for n, d in G2.degree()}

hub1 = max(dg1, key=dg1.get)
hub2 = max(dg2, key=dg2.get)
#node1 = min(dg1, key=dg1.get)
#node1 = np.random.randint(2*n)
node1 = 3*n

G4 = nx.compose(G1, G2)
G = nx.compose(G4, G3)
G.add_node(node1)
G.add_edge(node1, hub1)
G.add_edge(node1, hub2)
G.add_edge(node1, np.random.randint(2*n, 3*n))
#G.add_edge(node1)
#G.add_edge(np.random.randint(n), np.random.randint(n, 2*n))
#G.add_edge(np.random.randint(n), np.random.randint(2*n, 3*n))
#G.add_edge(np.random.randint(n, 2*n), np.random.randint(2*n, 3*n))
#G.add_edge(np.random.randint(n, 2*n), np.random.randint(2*n, 3*n))
#G.remove_edge(hub1, hub2)

#matrix_latex = create_latex_matrix(matrix)

## plot this shit with betweenness centrality and so on as color codes!!!
#G = nx.barabasi_albert_graph(n, 3)
#bc = nx.betweensess_centrality(G)
#dg = {n:d for n, d in G.degree()}
#cc = nx.closeness_centrality(G)
#ec = nx.eigenvector_centrality(G)
node_labels = {i+1 for i in range(len(G.nodes()))}
node_colors = [0.2 if i  < n else 0.4 if i < 2*n else 0.6 if i < 3*n else 0.9 for i in range(3*n + 1)]

pos = nx.drawing.nx_agraph.graphviz_layout(G, prog = 'neato')
fig = plt.figure(frameon=False, dpi=1200, figsize=(6,4))



#pos[3] = (300, 360)
color_blueish = '#56a891' #'#78c2ad'

color_blueish_lighter = '#bad6ce'
color_whitish = '#f8f9fa'

#node_color = [color_blueish if i in [3, 6, 10, 14] else color_blueish_lighter for i in range(3*n)]
#cmap = plt.Colormap(78c2ad)
nx.draw_networkx_nodes(G, pos, 
                       #node_color=list(ec.values()), 
                       node_color = node_colors,
                       vmin = 0,
                       vmax = 1,
                       cmap='Purples', 
                       node_size = 250
                       )
# nx.draw_networkx_edges(G, pos, edgelist = widths.keys(),
 #                       width=list(widths.values()))
nx.draw_networkx_edges(G, pos, width = 0.6, alpha = 0.25, edgelist=None)
#nx.draw_networkx_edges(G, pos, width = 0.6, alpha = 1, edgelist=[(3,6),  (6,10), (14, 10), (14, 3)])
#nx.draw_networkx_labels(G, pos,  font_color='white')
plt.show()

#fig.savefig('/Users/klaudiamur/Google Drive/Commwork/Pitch, Introduction documents/graphs/Pitch_presentation/broker.png', transparent=True)






    

    

    