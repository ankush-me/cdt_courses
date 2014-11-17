import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

H = sio.loadmat('H.mat')
H = H['history']
H = H[:10,:]

[N,P] = H.shape

ix = np.arange(N)
ip = np.arange(P)
[xx,pp] = np.meshgrid(ix,ip)
xp = np.c_[xx.flatten(),pp.flatten()].T
i_flat = np.ravel_multi_index(xp, H.shape)

G = nx.Graph()
G.add_nodes_from(i_flat)

for vi in i_flat:
	ix,ip = np.unravel_index(vi, H.shape)
	ix_p, ip_p = ix-1, H[ix,ip]-1
	if ix_p < 0:
		continue
	ravel_p = np.ravel_multi_index([[ix_p], [ip_p]], H.shape)
	G.add_edge(vi, int(ravel_p))

v_pos = {}
for vi in i_flat:
	ix,ip = np.unravel_index(vi,H.shape)
	v_pos[vi] = ip,H.shape[1]-ix

## get the list of edges which trace back from the last-row:
for ip in np.arange(P)



nx.draw_networkx_nodes(G,v_pos,node_size=50)
nx.draw_networkx_edges(G,v_pos,alpha=1.0,width=2)
plt.axis('off')
plt.savefig("house_with_colors.png") # save as png
plt.show() # display
