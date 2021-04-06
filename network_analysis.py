#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:16:20 2021

@author: klaudiamur
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd



df=pd.read_csv('/Users/klaudiamur/Box/Thesis/Datasets/dataverse_files/communication.csv', sep=';')

senders = np.sort(pd.unique(df['Sender']))
recipients = np.sort(pd.unique(df['Recipient']))

if np.array_equal(senders, recipients):
    users = senders
else:
    users = np.unique(np.concatenate((senders, recipients)))
    users = np.sort(users)
    
df['EventDate']= pd.to_datetime(df['EventDate'])
#### make a np adjacency matrix for every day

tmax = max(df['EventDate'])
date_max = tmax.date()
tmin = min(df['EventDate'])
date_min = tmin.date()
timespan = date_max - date_min
timespan = timespan.days + 1 

adj_matrix = np.zeros((len(users) + 1, len(users) + 1, timespan), dtype = int)

for i in range(len(df)):
    s = df.iloc[i, 0]
    r = df.iloc[i, 1]
    date = df.iloc[i, 2].date()
    #print()
    
    k = int((date - date_min).days)
    
    adj_matrix[s][r][k] += 1
    
adj_matrix_tot = np.sum(adj_matrix, axis = 2)

G = nx.from_numpy_matrix(adj_matrix_tot, create_using=nx.DiGraph)
    
plt.figure(frameon=False, dpi=500, figsize=(6,4))
pos = nx.drawing.nx_agraph.graphviz_layout(G, prog = 'neato')
nx.draw_networkx_nodes(G, pos, 
                       #node_color=list(ec.values()), 
                      # node_color = node_colors,
                       #vmin = 0,
                       vmax = 1,
                       cmap=plt.cm.Purples, node_size = 50
                       )
nx.draw_networkx_edges(G, pos, alpha = 0.8)

plt.show()



def plot_degree_dist(G):
    degrees = [G.out_degree(n) for n in G.nodes()]
    plt.hist(degrees)
    plt.show()

plot_degree_dist(G)

bc = nx.betweenness_centrality(G)
                     