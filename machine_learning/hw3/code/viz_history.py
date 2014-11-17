import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

H = sio.loadmat('H.mat')
H = H['history']
H = H[:20,:]

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
	#v_pos[vi] = H.shape[1]-ix, ip
	v_pos[vi] = ix, ip


## get the list of edges which trace back from the last-row:
active_edges = set()
active_nodes = set()
ix_src, ix_tar = 0, N-1
for ip_src in np.arange(P):
	for ip_tar in np.arange(P):
		i_src, i_tar = np.ravel_multi_index([[ix_src, ix_tar], [ip_src, ip_tar]], H.shape)
		if nx.has_path(G,i_src,i_tar):
			path = nx.shortest_path(G,i_src,i_tar)
			for n in path: active_nodes.add(n)
			for ipath in xrange(len(path)-1):
				active_edges.add((path[ipath],path[ipath+1]))

nx.draw_networkx_nodes(G,v_pos,node_size=50)
nx.draw_networkx_nodes(G,v_pos,node_size=50, nodelist=active_nodes, node_color='g')
nx.draw_networkx_edges(G,v_pos,alpha=0.2,width=2)
nx.draw_networkx_edges(G,v_pos,alpha=1.0,width=3, edgelist=active_edges)
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.ylabel('particles')
plt.xlabel('$z_n$ (class of $x_n$)')
plt.title('Sequential Monte-Carlo Ancestor Graph')
plt.savefig("smc_ancestry_vert.pdf") # save as pdf
plt.show() # display
