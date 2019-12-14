
# coding: utf-8

# In[1]:


import networkx as nx


# In[112]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle


# In[411]:


N = 10000
m = 5
G = nx.barabasi_albert_graph(N, m)
#plt.show(G)
#nx.draw(G)


# In[412]:


# color these motherfuckers ? 

greedy_prob = 3333
fair_prob = 3333
weak_prob = 3334


# In[413]:


strategy_dist = ["Greedy"]*greedy_prob + ["Fair"]*fair_prob + ["Weak"]*weak_prob
shuffle(strategy_dist)
arr = np.reshape(strategy_dist, (1, N))


# In[414]:


for index in range(0,N):
    G.node[index]['Strategy'] = arr[0][index]
    G.node[index]['Score'] = 0


# In[415]:


dict(G.node)


# In[416]:


game_matrix = np.array([[0, 0, 6], [0, 5, 5], [4, 4, 4]])

def game(graph):
    greedy_count = 0
    fair_count = 0
    weak_count = 0
    for i in range(0, N):
        for neighbor in list(graph.neighbors(i)):
            if graph.node[i]['Strategy'] == 'Greedy' and graph.node[neighbor]['Strategy'] == 'Greedy':
                graph.node[i]['Score'] += game_matrix[0,0]
            elif graph.node[i]['Strategy'] == 'Greedy' and graph.node[neighbor]['Strategy'] == 'Fair':
                graph.node[i]['Score'] += game_matrix[0,1]
            elif graph.node[i]['Strategy'] == 'Greedy' and graph.node[neighbor]['Strategy'] == 'Weak':
                graph.node[i]['Score'] += game_matrix[0,2]
            elif graph.node[i]['Strategy'] == 'Fair' and graph.node[neighbor]['Strategy'] == 'Greedy':
                graph.node[i]['Score'] += game_matrix[1,0]
            elif graph.node[i]['Strategy'] == 'Fair' and graph.node[neighbor]['Strategy'] == 'Fair':
                graph.node[i]['Score'] += game_matrix[1,1]
            elif graph.node[i]['Strategy'] == 'Fair' and graph.node[neighbor]['Strategy'] == 'Weak':
                graph.node[i]['Score'] += game_matrix[1,2]
            elif graph.node[i]['Strategy'] == 'Weak' and graph.node[neighbor]['Strategy'] == 'Greedy':
                graph.node[i]['Score'] += game_matrix[2,0]
            elif graph.node[i]['Strategy'] == 'Weak' and graph.node[neighbor]['Strategy'] == 'Fair':
                graph.node[i]['Score'] += game_matrix[2,1]
            elif graph.node[i]['Strategy'] == 'Weak' and graph.node[neighbor]['Strategy'] == 'Weak':
                graph.node[i]['Score'] += game_matrix[2,2]
                
            if graph.node[neighbor]['Score'] > graph.node[i]['Score']:
                graph.node[i]['Strategy'] = graph.node[neighbor]['Strategy']
                
            # try this with a best choice model - trying your neighbor's strategy and choosing if it would work best 
            # for you 
            
        if graph.node[i]['Strategy'] == 'Greedy':
                greedy_count += 1
        elif graph.node[i]['Strategy'] == 'Fair':
                fair_count += 1
        elif graph.node[i]['Strategy'] == 'Weak':
                weak_count += 1
    print(greedy_count, fair_count, weak_count)  


# In[418]:


game(G)


# In[353]:


def game_best_choice(graph):
    greedy_count = 0
    fair_count = 0
    weak_count = 0
    for i in range(0, N):
        for neighbor in list(graph.neighbors(i)):
            if graph.node[i]['Strategy'] == 'Greedy' and graph.node[neighbor]['Strategy'] == 'Greedy':
                graph.node[i]['Score'] += game_matrix[0,0]
            elif graph.node[i]['Strategy'] == 'Greedy' and graph.node[neighbor]['Strategy'] == 'Fair':
                graph.node[i]['Score'] += game_matrix[0,1]
            elif graph.node[i]['Strategy'] == 'Greedy' and graph.node[neighbor]['Strategy'] == 'Weak':
                graph.node[i]['Score'] += game_matrix[0,2]
            elif graph.node[i]['Strategy'] == 'Fair' and graph.node[neighbor]['Strategy'] == 'Greedy':
                graph.node[i]['Score'] += game_matrix[1,0]
            elif graph.node[i]['Strategy'] == 'Fair' and graph.node[neighbor]['Strategy'] == 'Fair':
                graph.node[i]['Score'] += game_matrix[1,1]
            elif graph.node[i]['Strategy'] == 'Fair' and graph.node[neighbor]['Strategy'] == 'Weak':
                graph.node[i]['Score'] += game_matrix[1,2]
            elif graph.node[i]['Strategy'] == 'Weak' and graph.node[neighbor]['Strategy'] == 'Greedy':
                graph.node[i]['Score'] += game_matrix[2,0]
            elif graph.node[i]['Strategy'] == 'Weak' and graph.node[neighbor]['Strategy'] == 'Fair':
                graph.node[i]['Score'] += game_matrix[2,1]
            elif graph.node[i]['Strategy'] == 'Weak' and graph.node[neighbor]['Strategy'] == 'Weak':
                graph.node[i]['Score'] += game_matrix[2,2]
            
    # try this with a best choice model - trying your neighbor's strategy and choosing if it would work best 
    # for you 
    if graph.node[i]['Score'] < 
        
        
        
        if graph.node[i]['Strategy'] == 'Greedy':
                greedy_count += 1
        elif graph.node[i]['Strategy'] == 'Fair':
                fair_count += 1
        elif graph.node[i]['Strategy'] == 'Weak':
                weak_count += 1
    print(greedy_count, fair_count, weak_count) 

