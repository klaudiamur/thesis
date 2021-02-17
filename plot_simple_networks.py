#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:00:11 2021

@author: klaudiamur
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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
    widths = nx.get_edge_attributes(G, 'weight')
    
    plt.figure(frameon=False, dpi=500, figsize=(6,4))
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog = 'neato')
    nx.draw_networkx_nodes(G, pos, node_color='Brown', node_size = 250)
   # nx.draw_networkx_edges(G, pos, edgelist = widths.keys(),
    #                       width=list(widths.values()))
    nx.draw_networkx_edges(G, pos, width = 0.5)
    nx.draw_networkx_labels(G, pos)
    plt.show()

    
n = 8
#matrix, G = create_random_network(n, directed = False, weighted = True)
matrix, G, b = create_bridge(n)
draw_network(G)

matrix_latex = create_latex_matrix(matrix)

## plot this shit with betweenness centrality and so on as color codes!!!

    

    

    