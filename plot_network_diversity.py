#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 14:49:01 2021

@author: klaudiamur
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import datetime
import random

def isNaN(num):
    return num != num

def make_network(df): ###users are the total list of users, df contains ts, user from and user to
  
    users = pd.unique(pd.concat([df['Sender'], df['Recipient']]))
    users = np.sort(users)
    
    time_min = min(df['EventDate']).date()
    time_max = max(df['EventDate']).date()
    
    d = +datetime.timedelta(days = 1)

    t = time_min

    timescale = []
    while t <= time_max:
        t = t + d
        timescale.append(t)
        
    duration = len(timescale)

    network = np.zeros((len(users), len(users), duration), dtype=int)

    for k in range(len(df)):
        sender = df.iloc[k]['Sender']
        recipient = df.iloc[k]['Recipient']
        time = df.iloc[k]['EventDate'].date()

        i = np.where(users == sender)[0][0]
        j = np.where(users == recipient)[0][0]
        t = (time-time_min).days

        network[i, j, t] = network[i, j, t] + 1

    return {'nw': network, 'ts': timescale}, users

def make_time_network(network, ts, timescale): 
    ### make weekly timescale network out of daily timescale network
    
    if timescale == 'week': 
        time_ts = [i.isocalendar()[1] for i in ts]
    elif timescale == 'month':
        time_ts = [i.month for i in ts]
    else:
        raise ValueError('The timescale has to be week or month')
        

    u, indices = np.unique(time_ts, return_inverse=True)
    ### sum up the network based on the number in weekly_ts!
    time_network = np.zeros((np.shape(network)[0], np.shape(network)[1], len(u)))

  
    for i in list(range(len(u))):
        indx = np.nonzero(i == indices)[0]
        new_network = np.sum(network[:, :, indx], axis = 2)
        time_network[:, :, i] = new_network
        #weekly_network = np.stack(weekly_network, new_network)

    return {'nw': time_network, 'ts': u}


def draw_network(G):
    widths = nx.get_edge_attributes(G, 'weight')
    
    plt.figure(frameon=False, dpi=500, figsize=(10,10))
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog = 'neato')
    nx.draw_networkx_nodes(G, pos, node_color='Brown', node_size = 50)
   # nx.draw_networkx_edges(G, pos, edgelist = widths.keys(),
    #                       width=list(widths.values()))
    nx.draw_networkx_edges(G, pos, width = 0.5, alpha = 0.6)
    #nx.draw_networkx_labels(G, pos)
    plt.show()
    

#### take only connections that have communicated at least 5 times in both directions!
def make_graph(network, ts):
    ts = 4 # threshold for number of connections
    nw_cleared = np.zeros_like(network)
    for i in range(len(network)):
        for j in range(i+1, len(network)):
            if network[i][j] > ts and network[j][i] > ts:
                nw_cleared[i][j] = 1
                nw_cleared[j][i] = 1
                            
            ### make graph out of nw_cleared (only subgraphs?)
    G = nx.from_numpy_matrix(nw_cleared)
    G.remove_nodes_from(list(nx.isolates(G))) # remove isolates            
    
    active_users = [i for i in G.nodes] 
    
    return G, active_users


#nw_data = np.genfromtxt('/Users/klaudiamur/Box/Thesis/Datasets/dataverse_files/communication.csv',delimiter=';')

nw_data = pd.read_csv('/Users/klaudiamur/Box/Thesis/Datasets/dataverse_files/communication.csv', sep=';')
nw_data['EventDate'] = pd.to_datetime(nw_data['EventDate'])

nw, users = make_network(nw_data)

formal_hierarchy_data = pd.read_csv('/Users/klaudiamur/Box/Thesis/Datasets/dataverse_files/reportsto.csv', sep=';')
formal_hierarchy_nw = np.zeros((len(users), len(users)), dtype=int)

formal_hierarchy_data['ID'] = pd.to_numeric(formal_hierarchy_data['ID'], errors='coerce', downcast='integer')
formal_hierarchy_data['ReportsToID'] = pd.to_numeric(formal_hierarchy_data['ReportsToID'], errors='coerce', downcast='signed')

for k in range(len(formal_hierarchy_data)):
    i_usr = formal_hierarchy_data.iloc[k]['ID']
    j_usr = formal_hierarchy_data.iloc[k]['ReportsToID']
        
    if not isNaN(j_usr):
        j_usr = int(j_usr)
        i = np.where(users == i_usr)[0][0]
        j = np.where(users == j_usr)[0][0]
        if not i == j:
            formal_hierarchy_nw[j][i] = 1
        
formal_hierarchy_tree = nx.from_numpy_matrix(formal_hierarchy_nw, create_using=nx.DiGraph())
formal_hierarchy_tree.remove_nodes_from(list(nx.isolates(formal_hierarchy_tree)))


draw_network(formal_hierarchy_tree)

#### find boss/hierarchy:
boss_indx = 0

for i in range(len(formal_hierarchy_nw)):    
    if np.sum(formal_hierarchy_nw, axis = 0)[i] == 0:
        if np.sum(formal_hierarchy_nw, axis = 1)[i] > 0:
            boss_indx = i
            
#boss = users[boss_indx]
## ok der rest isch oanfoch distance von boss? easy!

hierarchy= dict(nx.single_source_shortest_path_length(formal_hierarchy_tree, boss_indx))


nx.is_directed_acyclic_graph(formal_hierarchy_tree)
hierarchy = list(nx.topological_sort(formal_hierarchy_tree))
 ### all the employees
hierarchy_layers = [[k for k, v in hierarchy.items() if v == i] for i in range(max(hierarchy.values()) + 1)]



###now use just those who actually communicate and plot both networks on top of each other
### ok wait, make a list of managers here!!
### find branch of whole network for plotting!

subtrees = {m:dict(nx.single_source_shortest_path_length(formal_hierarchy_tree, m)) for m in hierarchy_layers[1]}

len_subtrees = {k:len(v) for k, v in subtrees.items()}
depth_subtrees = {k:max(v.values()) for k, v in subtrees.items()}

sub_subtrees = {m:dict(nx.single_source_shortest_path_length(formal_hierarchy_tree, m)) for m, v in subtrees[68].items() if v == 1 }

# =============================================================================
# plot formal and informal network strucutre (with 1 subtree)
# =============================================================================
start_node = 68

#subtree_dict = sub_subtrees[start_node].keys()
subtree_dict = subtrees[start_node]
subtree_nodes = list(subtree_dict.keys())
#subtree_nodes = list(sub_subtrees[start_node].keys())
G_hier = formal_hierarchy_tree.subgraph(subtree_nodes)
nw_tot = np.sum(nw['nw'], axis = 2)
G, active_users = make_graph(nw_tot, 2)
G_comm = G.subgraph(subtree_nodes)

dg = {n:d for n, d in G_comm.degree()}
cc = nx.closeness_centrality(G_comm)
ec = nx.eigenvector_centrality(G_comm)
bc = nx.betweenness_centrality(G_comm)
max_depth = max(subtree_dict.values())
node_colors_dict = {k:(1.1-v/max_depth) for k, v in subtree_dict.items()}
node_colors = [node_colors_dict[k] for k in G_hier.nodes]

node_colors = [bc[k] if k in bc.keys() else 0 for k in G_hier.nodes]


pos = nx.drawing.nx_agraph.graphviz_layout(G_hier, prog = 'neato')
pos1 = {k:v for k, v in pos.items() if k in G_comm.nodes}

fig = plt.figure(frameon=False, dpi=500, figsize=(10,10))

    #pos = nx.drawing.nx_agraph.graphviz_layout(G, prog = 'neato')
nx.draw_networkx_nodes(G_hier, pos, 
                       #node_color='Purple', 
                       cmap = 'Purples',
                       #vmax = 1.2,
                       vmin = -0.05,
                       node_color = node_colors,
                       node_size = 120)
nx.draw_networkx_edges(G_hier, pos, alpha = 0.8)
                       #edgelist = widths.keys(),                       
                       #width=list(widths.values()))
nx.draw_networkx_edges(G_comm, pos1, width = 0.4, alpha = 0.6)

    #nx.draw_networkx_labels(G, pos)

plt.show()
fig.savefig('/Users/klaudiamur/Google Drive/Commwork/Pitch, Introduction documents/graphs/Pitch_presentation/formal_informal_betweenness_centrality.png', transparent=True)



# =============================================================================
# plot the "real" subtree (just comms structure)
# ok fuck it it is not beautiful
# =============================================================================

node_colors = [node_colors_dict[k] for k in G_comm.nodes]

pos1 = nx.drawing.nx_agraph.graphviz_layout(G_comm, prog = 'neato')
fig = plt.figure(frameon=False, dpi=500, figsize=(10,10))
nx.draw_networkx_nodes(G_comm, pos1, 
                       cmap = 'Purples',
                       node_color= node_colors,
                       vmax = 1.1,
                       vmin = -0.5,
                       #node_color='Purple', 
                       node_size = 120)
#nx.draw_networkx_edges(G_hier, pos, alpha = 0.8)
                       #edgelist = widths.keys(),                       
                       #width=list(widths.values()))
nx.draw_networkx_edges(G_comm, pos1, width = 0.5, alpha = 0.6)

    #nx.draw_networkx_labels(G, pos)

plt.show()



# =============================================================================
# make labels for nodes (male/female, whatever) based on distance from boss
# ok let's do it with a different network though!
# =============================================================================

managers = [k for k, v in subtree_dict.items() if v < 2 and k in G_comm.nodes]

max_ec = max(ec.values())
gender_num = {k:(v/max_ec + random.uniform(-1, 1)) for k, v in ec.items()} ## negative proportional to ec! (let's see what happens)
gender = {k:('f' if v < 0.5 else 'm') for k, v in gender_num.items()}

node_colors = ['g' if k in managers else 'r' if gender[k] == 'f' else 'b' for k in G_comm.nodes]

pos1 = nx.drawing.nx_agraph.graphviz_layout(G_comm, prog = 'neato')
fig = plt.figure(frameon=False, dpi=500, figsize=(10,10))
nx.draw_networkx_nodes(G_comm, pos1, 
                       cmap = 'Purples',
                       node_color= node_colors,
                       vmax = 1.1,
                       vmin = -0.5,
                       #node_color='Purple', 
                       node_size = 120)
#nx.draw_networkx_edges(G_hier, pos, alpha = 0.8)
                       #edgelist = widths.keys(),                       
                       #width=list(widths.values()))
nx.draw_networkx_edges(G_comm, pos1, width = 0.5, alpha = 0.6)

    #nx.draw_networkx_labels(G, pos)

plt.show()


#### find average over how many females/males there are in the company in total and over how many the managers see

managers_egonetworks = [[gender[k] for k in list(nx.ego_graph(G_comm, i).nodes) ] for i in managers]
av_managers_enw = [[1 if i == 'f' else 0 for i in j] for j in managers_egonetworks]
average_egonw = [np.mean(i) for i in av_managers_enw]

averg_managers_tot = np.mean(average_egonw)

averg_real = [1 if v == 'f' else 0 for v in gender.values()]
averg_real_tot = np.mean(averg_real)


plt.pie([averg_real_tot, 1- averg_real_tot], colors = ['r', 'b'])

plt.pie([averg_managers_tot, 1-averg_managers_tot], colors = ['r', 'b'])



# =============================================================================
# 
# =============================================================================



nw_tot = np.sum(nw['nw'], axis = 2)

time_nw = make_time_network(nw['nw'], nw['ts'], 'month')

nw_march = time_nw['nw'][:, :, 2]

G1, active_users = make_graph(nw_march, 4)
     
formal1 = formal_hierarchy_tree.subgraph(active_users)

draw_network(formal1)

draw_network(G1)



pos = nx.drawing.nx_agraph.graphviz_layout(formal_hierarchy_tree, prog = 'neato')
pos1 = {k:v for k, v in pos.items() if k in active_users}

plt.figure(frameon=False, dpi=500, figsize=(10,10))

    #pos = nx.drawing.nx_agraph.graphviz_layout(G, prog = 'neato')
nx.draw_networkx_nodes(G1, pos1, node_color='Brown', node_size = 50)
   # nx.draw_networkx_edges(G, pos, edgelist = widths.keys(),
    #                       width=list(widths.values()))
nx.draw_networkx_edges(G1, pos1, width = 0.5, alpha = 0.6)

    #nx.draw_networkx_labels(G, pos)
plt.show()



# =============================================================================
# ### pick the ones with high closeness centrality as managers (or just use the real managers?)
# and for the rest a function of being male/female based on closeness?
# fuck it let's pick a part of the network tomorrow!
# =============================================================================


