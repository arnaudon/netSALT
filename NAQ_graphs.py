# This file is part of NAQ-graphs.
#
# Copyright (C) 2019, Alexis Arnaudon (alexis.arnaudon@imperial.ac.uk), 
# https://github.com/ImperialCollegeLondon/NAQ-graphs.git
#
# NAQ-graphs is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# NAQ-graphs is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NAQ-graphs.  If not, see <http://www.gnu.org/licenses/>.
#

import sys as sys
import numpy as np
import scipy as sc
import scipy.optimize as opt
import pickle as pickle

import pylab as plt
import networkx as nx
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

class NAQ(object):
    """ 
    This is the main class describing non-abelian quantum graphs
    """

    def __init__(self, G , positions = None, lengths = None, tot_len = 0, chi = None , group = 'U1', open_graph = False, transport_graph = False):
        
        #type method for finding the spectrum:
        # svd: use smallest singular value
        # eig: use smallest eigenvalue 
        # cond: use condition number 
        self.cond_type = 'svd' 

        self.n_processes_scan = 2

        if G.is_directed():
            print('Error, the graph must be undirected!')
        else:
            self.graph = G
            
        self.n = len(self.graph.nodes()) #number of nodes
        self.m = len(self.graph.edges()) #number of edges

        self.tot_len = tot_len #total length of inner edges, if 0, the lengths won't be rescaled

        self.group = group #set the group type
        self.open_graph = open_graph

        if self.open_graph:
            self.pump_params = None

        self.transport_graph = transport_graph
        if self.transport_graph:
            self.input_nodes = []
            self.output_nodes = []
            self.input_values = []

        if self.group == 'U1': #Abelian graph
            self.dtype = 'complex64' #float or complex numbers
            self.dim = 1  #dimension of the lie algebra/representation

        elif self.group == 'O3': #rotation group
            self.dtype = 'float64'
            self.dim = 3  #dimension of the lie algebra/representation

        elif self.group == 'SU3':
            self.dtype = 'float64'
            print('Do not try this, not implemented yet!')

        else:
            print('Which group? I do not know this one!')


        #pre-compute the mask for inner edges 
        self.in_mask = sc.sparse.lil_matrix((2*self.m, 2*self.m))
        for ei, e in enumerate(list(self.graph.edges())):
            if len(self.graph[e[0]])>1 and len(self.graph[e[1]])>1:
                self.in_mask[2*ei,2*ei] = 1
                self.in_mask[2*ei+1,2*ei+1] = 1
                  

        #if we know the position, we can set the corresponding lengths
        if positions:
            self.pos = positions
            for u in self.graph.nodes():
                self.graph.node[u]['pos'] = self.pos[u] #set the position to the networkx graph

            lengths = np.zeros(self.m)
            for ei, e in enumerate(list(self.graph.edges())):
                (u, v) = e[:2]
                lengths[ei]   = sc.linalg.norm(self.graph.node[u]['pos'] - self.graph.node[v]['pos']) #set the length  

            self.set_lengths(lengths)

        else: #otherwise just use force atlas for plotting the graphs 
            print("Using force atlas to set node positions")
            from fa2 import ForceAtlas2
            forceatlas2 = ForceAtlas2(
                        # Tuning
                        scalingRatio=2.,
                        strongGravityMode=False,
                        gravity=1.0,
                        outboundAttractionDistribution=False,  # Dissuade hubs
                        # Log
                        verbose=False)

            self.pos = forceatlas2.forceatlas2_networkx_layout(self.graph, pos=None, iterations=2000)

            for u in self.graph.nodes():
                self.graph.node[u]['pos'] = self.pos[u] #set the position to the networkx graph

            #if we know the lengths, we set it directly
            if lengths:
                self.set_lengths(lengths)
            
            #if none are set, just set lengths to 1
            else:
                self.set_lengths(np.ones(self.m))

        #set the chi wavenumber 
        if chi is not None:
            self.set_chi(chi)
            self.chi0 = self.chi.copy() #save the original chi for later
        else:
            print('Please provide an edge generator')

    def set_lengths(self, lengths):
        #set the lengths of the edges
        
        #if a total length has be set, rescale the lengths accordingly 
        actual_tot_len = self.in_mask.todense()[::2,::2].dot(lengths).sum()
        n_inner = np.shape(np.where( np.diag(self.in_mask.todense()[::2,::2])>0) )[1]


        if self.tot_len > 0:
            lengths = list(np.array(lengths)*self.tot_len/actual_tot_len)
            print("Total lenght:",  self.tot_len)
            print("Average lenght:",  self.tot_len/n_inner)
        else:
            print("Total lenght:",  actual_tot_len)
            print("Average lenght:",  actual_tot_len/n_inner)



        self.lengths  = lengths
        for ei, e in enumerate(list(self.graph.edges())):
            (u, v) = e[:2]
            self.graph[u][v]['L']  = lengths[ei]

    

    def set_chi(self, chi):
        #set the chi variable on the edges
        self.chi = chi
        for ei, e in enumerate(list(self.graph.edges())):
            (u, v) = e[:2]
            self.graph[u][v]['chi']  = chi[ei]

    def update_chi(self, mode):
        #set edge wavenumbers

        chi = self.chi0.copy()

        if isinstance(mode, (list, np.ndarray) ):  #complex wavenumber 
            k = mode[0]-1.j*mode[1]
        else:  #real wavenumber
            k = mode

        #if a pump is set
        if self.open_graph and self.pump_params is not None:
            self.gamma = self.pump_params['gamma_perp'] / ( k - self.pump_params['k_a'] + 1.j * self.pump_params['gamma_perp'])
            self.pump_mask = sc.sparse.lil_matrix((2*self.m, 2*self.m))
            for e in self.pump_params['edges']:
                chi[e] *= np.sqrt(1. + self.gamma * self.pump_params['D0'])
                self.pump_mask[2*e,2*e] = 1.
                self.pump_mask[2*e+1,2*e+1] = 1.

        self.set_chi(k*chi) #set the new chi

    def Winv_matrix(self):
        """
        Construct the matrix W^{-1}
        """

        if self.group == 'U1':
            self.Winv_matrix_U1()

        if self.group == 'O3':
            self.Winv_matrix_O3()
            
    def diag_chi(self):
        """
        Compute the diag (chi)
        """
        if self.group == 'U1':

            Chi = sc.sparse.lil_matrix(( 2 * self.m, 2 * self.m), dtype = self.dtype) 

            for ei, e in enumerate(list(self.graph.edges())):
                (u, v) = e[:2]
                Chi[2*ei, 2*ei] = self.graph[u][v]['chi']
                Chi[2*ei+1, 2*ei+1] = self.graph[u][v]['chi']
        
            self.Chi = Chi.asformat('csc')
        else:
            print('not yet implemented')
            
    def Winv_matrix_U1(self):
        """
        Construct the matrix W^{-1} for U1
        """
        
        Winv = sc.sparse.lil_matrix(( 2 * self.m, 2 * self.m), dtype = self.dtype) 
        
        for ei, e in enumerate(list(self.graph.edges())):
            (u, v) = e[:2]
                
            Winv[2*ei, 2*ei] = 1. / ( np.exp( 2.* self.graph[u][v]['L'] * self.graph[u][v]['chi'] ) - 1. ) 
            Winv[2*ei+1, 2*ei+1] = Winv[2*ei, 2*ei]

        self.Winv = Winv.asformat('csc')
       
    def Z_matrix_U1(self):
        """
        Construct the matrix Z for U1 (used for computing the pump overlapping factor)
        """
        
        Z = sc.sparse.lil_matrix(( 2 * self.m, 2 * self.m), dtype = self.dtype) 
        
        for ei, e in enumerate(list(self.graph.edges())):
            (u, v) = e[:2]
 
            Z[2*ei, 2*ei] =  (np.exp( 2.* self.graph[u][v]['L'] * self.graph[u][v]['chi'] ) - 1.)/(2.* self.graph[u][v]['chi'])
            Z[2*ei, 2*ei+1] = self.graph[u][v]['L']*np.exp( self.graph[u][v]['L'] * self.graph[u][v]['chi'] )
        
            Z[2*ei+1, 2*ei] = Z[2*ei, 2*ei+1]
            Z[2*ei+1, 2*ei+1] = Z[2*ei, 2*ei]

        self.Z = Z.asformat('csc')

    def Winv_matrix_O3(self):
        """
        Construct the matrix W^{-1} for O3
        """
        
        Winv = sc.sparse.lil_matrix(( self.dim * 2 * self.m, self.dim * 2 * self.m), dtype = self.dtype) 
        
        for ei, e in enumerate(list(self.graph.edges())):
            (u, v) = e[:2]
            ei *= self.dim 
 
            Winv[2*ei:2*ei+self.dim, 2*ei:2*ei+self.dim] = np.linalg.pinv( sc.linalg.expm( 2.* self.graph[u][v]['L'] * self.graph[u][v]['chi'] ) - np.eye(self.dim)).dot(self.graph[u][v]['chi'] )

            Winv[2*ei+self.dim:2*ei+2*self.dim, 2*ei+self.dim:2*ei+2*self.dim] = Winv[2*ei:2*ei+self.dim, 2*ei:2*ei+self.dim]

        self.Winv = Winv.asformat('csc')
        

    def Winv_matrix_O3_other(self):
        """
        Construct the matrix W^{-1} for O3
        """
        
        Winv = sc.sparse.lil_matrix(( self.dim * 2 * self.m, self.dim * 2 * self.m), dtype = self.dtype) 
        
        for ei, e in enumerate(list(self.graph.edges())):
            (u, v) = e[:2]
            ei *= self.dim 
 
            if not self.no_projection:

                chi = np.zeros(3)
                chi[0] = self.graph[u][v]['chi'][0,1]
                chi[1] = self.graph[u][v]['chi'][0,2]
                chi[2] = self.graph[u][v]['chi'][2,1]
                chi_norm = chi/np.linalg.norm(chi)

                Winv[2*ei:2*ei+self.dim, 2*ei:2*ei+self.dim] = np.linalg.pinv(sc.linalg.expm( 2.* self.graph[u][v]['L'] * self.graph[u][v]['chi'] ) - np.eye(self.dim)).dot(self.graph[u][v]['chi'] )/np.linalg.norm(chi)


                if self.unit_w:
                    eigs = np.abs(np.linalg.eig( sc.linalg.expm( 2.* self.graph[u][v]['L'] * self.graph[u][v]['chi'] ) - np.eye(self.dim))[0])
                    Winv[2*ei:2*ei+self.dim, 2*ei:2*ei+self.dim] *= np.max(eigs) 
                Winv[2*ei:2*ei+self.dim, 2*ei:2*ei+self.dim] += np.outer(chi_norm,chi_norm)/self.graph[u][v]['L'] 
            if self.no_projection:
                Winv[2*ei:2*ei+self.dim, 2*ei:2*ei+self.dim] = sc.linalg.expm( self.graph[u][v]['L'] *self.graph[u][v]['chi'] )
                if not self.unit_w:
                    eigs = np.abs(np.linalg.eig( sc.linalg.expm( 2.* self.graph[u][v]['L'] * self.graph[u][v]['chi'] ) - np.eye(self.dim))[0])
                    Winv[2*ei:2*ei+self.dim, 2*ei:2*ei+self.dim] /= np.max(eigs) 

        Winv[2*ei+self.dim:2*ei+2*self.dim, 2*ei+self.dim:2*ei+2*self.dim] = Winv[2*ei:2*ei+self.dim, 2*ei:2*ei+self.dim]

        self.Winv = Winv.asformat('csc')
        
 
    def B_matrices(self):
        """
        Construct the incidence matrix B, from nodes to edges
        """

        if self.group == 'U1':
            self.B_matrices_U1()

        if self.group == 'O3':
            self.B_matrices_O3()


    def B_matrices_U1(self):
        """
        Construct the incidence matrix B, from nodes to edges, for U1
        """
                
        B = sc.sparse.lil_matrix((self.dim * 2 * self.m, self.dim * self.n), dtype = self.dtype)
        
        for ei, e in enumerate(list(self.graph.edges())):
            (u, v) = e[:2]

            expl = np.exp(self.graph[u][v]['L'] * self.graph[u][v]['chi'])

            B[2 * ei, u] = -1.
            B[2 * ei, v] = expl 
    
            B[2 * ei + 1, u] = expl 
            B[2 * ei + 1, v] = -1.

        if self.open_graph or self.transport_graph: #only outgoing waves at the degree 1 nodes

            self.BT = (B.T).asformat('csc').copy() #save the transpose of B
            for ei, e in enumerate(list(self.graph.edges())):
                (u, v) = e[:2]
          
                if len(self.graph[u]) == 1:
                    B[2 * ei , v]     = 0.
                    B[2 * ei + 1 , u] = 0.
                        
                if len(self.graph[v]) == 1:
                    B[2 * ei + 1 , u] = 0.
                    B[2 * ei  , v] = 0.

            self.B = B.asformat('csc')

        else:
            self.B = B.asformat('csc')
            self.BT = (B.T).asformat('csc')
    

    def B_matrices_O3(self):
        """
        Construct the incidence matrix B, from nodes to edges, for O3
        """
        
        B = sc.sparse.lil_matrix((self.dim * 2 * self.m, self.dim * self.n), dtype = self.dtype)
        BT = sc.sparse.lil_matrix((self.dim * self.n, self.dim * 2 * self.m), dtype = self.dtype)
        
        for ei, e in enumerate(list(self.graph.edges())):
            (u, v) = e[:2]

            expl = sc.linalg.expm(self.graph[u][v]['L'] * self.graph[u][v]['chi'])

            ei *= self.dim 
            u  *= self.dim 
            v  *= self.dim 

            B[2 * ei:2*ei+self.dim, u:u+self.dim] = -np.eye(self.dim)
            B[2 * ei:2*ei+self.dim, v:v+self.dim] = expl 
    
            B[2 * ei + self.dim:2*ei+2*self.dim, u:u+self.dim] = B[2 * ei:2*ei+self.dim, v:v+self.dim]
            B[2 * ei + self.dim:2*ei+2*self.dim, v:v+self.dim] = B[2 * ei:2*ei+self.dim, u:u+self.dim]
        
            #here, we the transpose by hand, as it is only the partial transpose (not for chi) 
            BT[u:u+self.dim, 2 * ei:2*ei+self.dim] = -np.eye(self.dim)
            BT[v:v+self.dim, 2 * ei:2*ei+self.dim] =  expl 
    
            BT[u:u+self.dim, 2 * ei + self.dim:2*ei+2*self.dim] = BT[v:v+self.dim, 2 * ei:2*ei+self.dim]
            BT[v:v+self.dim, 2 * ei + self.dim:2*ei+2*self.dim] = BT[u:u+self.dim, 2 * ei:2*ei+self.dim]

        self.B = B.asformat('csc')
        self.BT = BT.asformat('csc')
    

    def node_laplacian(self):
        """
        Construct the L_0 Laplacian, from nodes to nodes
        """
        
        self.L0 = self.BT.dot(self.Chi.dot(self.Winv)).dot(self.B).asformat('csc')

    def update_laplacian(self):
        """
        Update the Laplacian with chi 
        """
        self.Winv_matrix()    #compute the inverse of W 
        self.B_matrices()     #compute the incidence matrix
        self.diag_chi()
        self.node_laplacian() #compute the node Laplacian
               
    def test_laplacian(self):
        """
        test the Laplacian, smallest singular value and rank
        """

        self.update_laplacian() #update the Laplacian with chi


        if not self.transport_graph:
            if self.group == 'U1':
                L = self.L0.todense()

            if self.cond_type == 'svd':
                try:
                    u, w, vh = np.linalg.svd(L)           #compute the new singular value
                    return np.min(abs(w))

                except np.linalg.LinAlgError:
                    print('svd error')
                    return 0.

            elif self.cond_type == 'eig':
                try:
                    w = sc.sparse.linalg.eigs(self.L0, k=1, sigma=0, return_eigenvectors=False)
                except np.linalg.LinAlgError:
                    print('eig error')
                    return 0.

                return np.min(abs(w))

            elif self.cond_type == 'cond':
                return -np.linalg.cond(L)

        if self.transport_graph:

            source = np.zeros(self.n)
            source[self.input_nodes] = self.input_values #construct the source vector
            solution = sc.sparse.linalg.spsolve(self.L0, source) #solve L * solution = source

            return abs(solution[self.output_nodes]).sum() # return the total flux on the output nodes


    def compute_spectrum(self, Ks, digit = 7):
        if self.group == 'U1':
            k_mu = self.compute_spectrum_U1(Ks, digit)

        if self.group == 'O3':
            k_mu = self.compute_spectrum_O3(Ks, digit)
        return k_mu

    def compute_spectrum_U1(self, Ks, digit):
        """
        Find the spectrum of the quantum graph, where solutions similar under digit digits are merged
        """

        def cond_L0(k):
            self.update_chi(k[0])
            return self.test_laplacian()

        k_mu = []
        for k in Ks:
            k_mu.append(sc.optimize.fmin(cond_L0, k, disp = False))

        return np.unique(np.around(k_mu, digit))

    def scan_k(self, Ks, Alphas = None):
        #scan the wavenumber k (possibly complex)

        if self.open_graph:
            if Alphas is not None:
                return self.scan_k_open(Ks, Alphas)
            else:
                print('Please provide an alpha vector')
        else:
            return self.scan_k_close(Ks)

    def scan_k_close(self, Ks):
        #scan the real wavenumber k
        s = np.zeros(len(Ks))
        for i,k in enumerate(Ks):
            self.update_chi([k,])
            s[i] = self.test_laplacian()
        return s

    def scan_k_open(self, Ks, Alphas):
        #scan the complex wavenumber k+i alpha

        Ks_list = []
        for i, k in enumerate(Ks):
            for j, alpha in enumerate(Alphas):
                Ks_list.append(np.array([k,alpha]))
                
        with Pool(processes = self.n_processes_scan) as p_scan:  #initialise the parallel computation
            out = list(tqdm(p_scan.imap(self.f_scan, Ks_list), total = len(Ks_list))) #run them 

        s = np.zeros([len(Ks),len(Alphas)] )
        kk = 0 
        for i in range(len(Ks)):
            for j in range(len(Alphas)):
                s[i,j] = out[kk]
                kk+=1
                
        max_s = np.zeros(len(Alphas))        
        for j in range(len(Alphas)):
            max_s[j] = np.max(s[:,j])
            s[:,j] = s[:,j] / max_s[j] #normalized by the largest singular value accross all real(k)
            
        return s, max_s #return the singular values and the normalisations

    def f_scan(self, k):
        #function to compute the scan in parallel
        self.update_chi(k)
        return self.test_laplacian()

    def plot_scan(self, ks, alphas, S, modes = None):
        """
        plot the scan with the mode found

        """
        plt.figure(figsize=(10,5))

        plt.imshow(np.log(S.T), extent= (ks[0], ks[-1], alphas[0], alphas[-1]), aspect='auto', origin='lower', vmax = 0, vmin = -3)

        cbar = plt.colorbar()
        cbar.set_label('smallest singular value')

        plt.xlabel(r'$Real(k)$')
        plt.ylabel(r'$\alpha = -Im(k)$')

        if modes is not None:
            plt.plot(modes[:,0], modes[:,1],'r+')

        plt.axis([ks[0], ks[-1], alphas[-1], alphas[0]])


    def compute_solution(self, s_min = 1e-2):
        """
        Compute the solution of L_0\phi = 0
        """
        if not self.transport_graph:

            if self.cond_type == 'svd' or self.cond_type == 'cond':
                try:
                    u, w, vh = np.linalg.svd(self.L0.todense())
                except np.linalg.LinAlgError:
                    print('svd error')

                s = np.min(w)
                v = vh[np.argmin(w)]

            elif self.cond_type == 'eig':
                w, v = sc.sparse.linalg.eigs(self.L0, k=1, sigma=0)
                s = np.min(abs(w))
                v = v[:,np.argmin(abs(w))]

            if s > s_min:
                print('Laplacian not singular!', s)
                return 0

            return np.array(v.conj()).flatten()

        if self.transport_graph:
            source = np.zeros(self.n)
            source[self.input_nodes] = self.input_values #construct the source vector
            solution = sc.sparse.linalg.spsolve(self.L0, source) #solve L * solution = source

            return solution # return the solution
    
    def compute_edge_mean_E2(self):
        """
        Compute the average |E|^2 on each edge
        """
        
        phi = self.compute_solution()
        flux = self.Winv.dot(self.BT.T).dot(phi) #we use BT.T as we need to make sure the the in-fluxes vanish
        
        edge_mean = np.zeros(self.m)
        for ei, e in enumerate(list(self.graph.edges())):
            (u, v) = e[:2]

            z = np.zeros([2,2])
            if abs(np.real(self.graph[u][v]['chi']))>0: #if k has a complex part (recall \chi = ik)
                z[0, 0] = 1.
            else: #we recast to real because it is real
                z[0, 0] = np.real((np.exp( self.graph[u][v]['L']*(self.graph[u][v]['chi'] + np.conj(self.graph[u][v]['chi'])) ) - 1.)/ (self.graph[u][v]['L']*(self.graph[u][v]['chi'] + np.conj(self.graph[u][v]['chi'])) ) )

            #no issue here if im(k)=0, and just recast to real 
            z[0, 1] =  np.real(( np.exp( self.graph[u][v]['L']*self.graph[u][v]['chi'] ) - np.exp( self.graph[u][v]['L']*np.conj(self.graph[u][v]['chi'])) ) / (self.graph[u][v]['L']*( self.graph[u][v]['chi'] - np.conj(self.graph[u][v]['chi']) ) ))
            
            #other matrix elements
            z[1, 0] = z[0, 1]
            z[1, 1] = z[0, 0]
            
            #then compute the norm
            edge_mean[ei] = np.real(np.conj(flux[2*ei:2*ei+2]).T.dot(z.dot(flux[2*ei:2*ei+2]))) #BUG flux est du mauvais type je crois, type(flux[2*ei:2*ei+2]): <class 'scipy.sparse.csc.csc_matrix'>
    
        return edge_mean
    
    def find_mode_brownian_ratchet(self, k_0, params, disp = True, save_traj = False):
        """
        Find the frequency and dissipation of the network, given an initial condition k
        
        """
 
        s_min = params['s_min']
        s_size = np.array(params['s_size'])
        max_steps = params['max_steps']
        reduc = params['reduc']

        k = k_0.copy()
        if save_traj: 
            K = []
            K.append(k.copy())

        self.update_chi(k)
        s = self.test_laplacian()

        s_0 = s
        n_wrong = 0
        N_steps = 0
        while s > s_min: #while the current singular value is larger then s_min, update

            dk = 1.- 2.*np.random.rand(2)         #increment K
            dk = s_size * s/s_0 * dk           #rescale it
            k  += dk                              #update theta
            
            if disp:
                print("Singular value:", s, "Step size:", s_size, k-dk,  k)

            self.update_chi(k)
            s_new = self.test_laplacian()
            
            # keep the update if the s_min is improved
            if s_new < s:
                s     =  s_new
                n_wrong = 0 
                    
                if save_traj: 
                    K.append(k.copy()) #save the Ks
            # if not, reset K to the previous step
            else: 
                k -= dk
                n_wrong += 1
        
            #if no improvements after 20 iteration, multiply the steps by reduc
            if n_wrong > 20:
                s_size *= reduc
                n_wrong = 0 
            
            #if the steps are too small, exit the loop
            if s_size[0] < 1e-5:
                break 
                    
            #if number of steps is too large, exit the loop
            N_steps +=1
            if N_steps > max_steps:
                k   -= dk 
                break 
            

        if s < s_min:
            #if the main loop exited normally, save the value
            if save_traj: 
                return np.array(K)
            else:
                return k
        else:
            if disp:
                print('did not find a solution!')
            return np.array([0,])
        
        
        
    def find_mode(self, k_0, params, max_s, step_Alpha, disp = False, save_traj = False): 
        """
        Find the frequency and dissipation of the network, given an initial condition k
        
        """
        s_min = params['s_min']
        s_size = np.array(params['s_size'])
        max_steps = params['max_steps']
        reduc = params['reduc']

        k = k_0.copy()
        if save_traj: 
            K = []
            K.append(k.copy())

        self.update_chi(k)
        s = self.test_laplacian()
        
        #the singular value is then normalized
        alpha = k[1]
        p = alpha/step_Alpha - int(alpha/step_Alpha)
        s /= ( (1-p)*max_s[int(alpha/step_Alpha)-1] + p*max_s[int(alpha/step_Alpha)]) #linear approximation of the normalisation coefficient
        
        s_0 = s
        s_new = s
        n_wrong = 0
        N_steps = 0
        alpha_sensibility = s_size[1]
        real_sensibility  = s_size[0]       
        while s > s_min: #while the current singular value is larger then s_min, update
            dk = alpha_sensibility * s/s_0 
            k_list_alpha = [k+[0,dk], k+[0,-dk], k] #up, down, original value
            result_alpha = [0, 0, s] 
            
            #fill in result_alpha with singular values
            for i in range(2):      
                #update theta
                k = k_list_alpha[i]
                
                if disp:
                    print("Singular value:", s, "Step size:", s_size, k-dk,  k)
    
                self.update_chi(k)
                s_new = self.test_laplacian()
            
                alpha = k[1]
                p = alpha/step_Alpha - int(alpha/step_Alpha)
                if alpha < 0:
                    s_new /= max_s[0]
                elif int(alpha/step_Alpha) >= len(max_s)-1: #if alpha goes beyond the range the range of alpha used to measure the matrix s, the normalisation coeff is taken at the limit
                    s_new /= max_s[-1]
                else: 
                    s_new /= ((1-p)*max_s[int(alpha/step_Alpha)] + p*max_s[int(alpha/step_Alpha)+1]) #linear approximation of the normalisation coefficient
                    
                #keep the update if the s_min is improved
                result_alpha[i] = s_new
                
            s = np.min(result_alpha) #[np.where(result_alpha == min(result_alpha))[0][0]]
            if s == result_alpha[2]: #if the best singular value did not change
                alpha_sensibility /= 2.
                n_wrong +=1
                
            #k = np.array(k_list_alpha[np.where(result_alpha == min(result_alpha))[0][0]]) #keep the value of k at which the singular value is the lowest
            k = np.array(k_list_alpha[np.argmin(result_alpha)]) #keep the value of k at which the singular value is the lowest

            dk = real_sensibility * s/s_0 
            k_list_real = [k+[dk,0], k+[-dk,0],k]
            result_real = [0, 0, s]

            for i in range(2):      
                #update theta
                k = k_list_real[i]
                if disp:
                    print("Singular value:", s, "Step size:", s_size, k-dk,  k)
    
                self.update_chi(k)
                s_new = self.test_laplacian()
            
                alpha = k[1]
                p = alpha/step_Alpha - int(alpha/step_Alpha)
                if alpha<0:
                    s_new /= max_s[0]
                elif int(alpha/step_Alpha)>=len(max_s)-1: #if alpha goes beyond the range the range of alpha used to measure the matrix s, the normalisation coeff is taken at the limit    
                    s_new /= max_s[-1]
                else: 
                    s_new /= ((1-p)*max_s[int(alpha/step_Alpha)]+p*max_s[int(alpha/step_Alpha)+1]) #linear approximation of the normalisation coefficient
                    
    #             keep the update if the s_min is improved
                result_real[i] = s_new
        
            s = np.min(result_real) #[np.where(result_real==min(result_real))[0][0]]
            
            if s == result_alpha[2]: #if the best singular value did not change
                real_sensibility /= 2
                n_wrong += 1
                
            #k = np.array(k_list_real[np.where(result_real==min(result_real))[0][0]]) #keep the value of k at which the singular value is the lowest
            k = np.array(k_list_real[np.argmin(result_real)]) #keep the value of k at which the singular value is the lowest

            if s_new < s:
                n_wrong = 0 
                    
                if save_traj: 
                    K.append(k.copy()) #save the Ks            
            
            #if no improvements after 20 iteration, multiply the steps by reduc
            if n_wrong > 20:
                s_size *= reduc
                n_wrong = 0 
            
            #if the steps are too small, exit the loop
            if s_size[0] < 1e-5:
                break 
                    
            #if number of steps is too large, exit the loop
            N_steps +=1
            if N_steps > max_steps:
                k   -= dk 
                break 
            

        if s < s_min:
            #if the main loop exited normally, save the value
            if save_traj: 
                return np.array(K)
            else:
                return k
        else:
            if disp:
                print('did not find a solution!')
            return np.array([0,])    

  
    def update_modes(self, modes, params):
        """
        Scan the new modes from a set of known modes, (for pump trajectories)
        """

        self.params = params 
        Ks_list = []
                
        with Pool(processes = self.n_processes_scan) as p_find:  #initialise the parallel computation
            out = p_find.map(self.f_find_brownian_ratchet, modes) #run them 
        
        #if we could not find a new mode, use the same as before
        new_modes = []
        for i, m in enumerate(out):
            if m[0]==0:
                new_modes.append(modes[i])
                print('Could not update a mode, use smaller D0 steps!')
            else:
                new_modes.append(m)

        return new_modes 
   
    

    def find_modes_brownian_ratchet(self, Ks, Alphas, params, th= 0.001):
        """
        Scan the singular values for solutions in K and A
        """
        self.params = params 
        Ks_list = []
        for i, k in enumerate(Ks):
            for j, alpha in enumerate(Alphas):
                Ks_list.append(np.array([k,alpha]))
        
        with Pool(processes = self.n_processes_scan) as p_find:  #initialise the parallel computation
            out = list(tqdm(p_find.imap(self.f_find_brownian_ratchet, Ks_list), total = len(Ks_list))) #run them 

        modes = []
        for m in out:
            if m[0] != 0:
                modes.append(m)

        modes = self.clean_modes(np.asarray(modes), th) #remove duplicates

        return modes[np.argsort(np.array(modes)[:,1])] #return sorted modes (by lossyness)
    
    
    def find_modes(self, Ks, Alphas, s, max_s, params, th= 0.01): #this function uses the result from the scanning to optimize mode search
        """
        Scan the singular values for solutions in K and A
        """
        
        self.params = params 
        Ks_list = []
        s_list = []
        for i, k in enumerate(Ks):
            for j, alpha in enumerate(Alphas):
                if s[i,j] < self.params['s_interest']: #zone of interest
                    Ks_list.append([k, alpha])
                    s_list.append(s[i,j])
                    
        Ks_list = np.array(Ks_list)[np.argsort(s_list)] #now that the list is sorted the information about s can be deleted
        #the list is sorted before it is cleaned so that the if multiple interest points are found to close to each others, 
        #the one with the smallest singular value is kept 
        Ks_list = self.clean_interest_points(Ks_list, params) 

        find_modef = partial(self.f_find, max_s, Alphas[1]-Alphas[0]) 
        with Pool(processes = self.n_processes_scan) as p_find:  #initialise the parallel computation
            out = list(tqdm(p_find.imap(find_modef, Ks_list), total = len(Ks_list))) #run them 

        modes = []
        for m in out:
            if m[0] != 0:
                modes.append(m)

        modes = self.clean_modes(np.asarray(modes), params) #remove duplicates

        return modes[np.argsort(np.array(modes)[:,1])] #return sorted modes (by lossyness)
    
   
    
    def f_find_brownian_ratchet(self,  k):
        #inner function to find modes in parallel
        return self.find_mode_brownian_ratchet(k, self.params, disp = False, save_traj = False)

    def f_find(self, max_s, step_alpha,  k):
        #inner function to find modes in parallel
        return self.find_mode(k, self.params, max_s, step_alpha, disp = False, save_traj = False)

    def clean_modes(self, modes, params):
        """
        Clean the modes obtained randomly to get rid of duplicates, with threshold th
        """
        s_size=params['s_size']
        id_selec  = [] #id of duplicated modes
        id_remove = [] #id of selected modes
        for i, m_all in enumerate(modes):
            if i not in id_remove:
                id_selec.append(i)
                for j, m_selec in enumerate(modes[i+1:]):
                    j+=(i+1) #if j not in id_remove: #to not remove all the modes
                    if j not in id_remove: #to not remove all the modes
                        if (abs((m_all-m_selec)[0]) < s_size[0]) and (abs((m_all-m_selec)[1]) < s_size[1]): #test if the same, and add the first mode to the list
                            id_remove.append(j)

        id_remove_unique = np.unique(id_remove)
        id_selec_unique, duplicates = np.unique(id_selec, return_index=False, return_counts=True)

        modes_clean = modes[id_selec_unique]

        print(len(modes_clean), "modes out of", len(modes), "attempts")

        return modes_clean
    
    
    def clean_interest_points(self, Ks_list, params):
        """
        Same function a clean_modes but for interest points
        """
        
        s_size = params['s_size']
        finesse = params['finesse']
        
        id_selec  = [] #id of duplicated mode
        id_remove = [] #id of selected modes
        for i, m_all in enumerate(Ks_list):
            if i not in id_remove:
                id_selec.append(i)
                for j, m_selec in enumerate(Ks_list[i+1:]):
                    j += i+1 #if j not in id_remove: #to not remove all the modes
                    if j not in id_remove: #to not remove all the modes
                        #test if the same, and add the first mode to the list
                        if abs((m_all-m_selec)[0]) < finesse*s_size[0] and abs((m_all-m_selec)[1]) < finesse*s_size[1]: 
                            id_remove.append(j)

        id_remove_unique = np.unique(id_remove)
        id_selec_unique, duplicates = np.unique(id_selec, return_index=False, return_counts=True)
        
        Ks_list_clean = []
        for j in  id_selec_unique:
            Ks_list_clean.append(Ks_list[j])

        return Ks_list_clean


    def pump_linear(self, mode, D0_0, D0_1):
            """
            find the linear approximation of the new wavenumber, for an original pump mode with D0_0, to a new pump D0_1
            """
                    
            #update the laplacian with this pump
            self.pump_params['D0'] = D0_0
            self.update_chi(mode)
            self.update_laplacian()
            
            #compute the node field
            phi = self.compute_solution()
            
            self.Z_matrix_U1() #compute the Z matrix
            edge_norm = self.Winv.dot(self.Z).dot(self.Winv) #compute the weight matrix for \int E^2
            
            #compute the inner sum
            L0_in = self.BT.dot(edge_norm.dot(self.in_mask)).dot(self.B).asformat('csc')
            L0_in_norm = phi.T.dot(L0_in.dot(phi))
            
            #compute the field on the pump
            L0_I = self.BT.dot(edge_norm.dot(self.pump_mask.dot(self.in_mask))).dot(self.B).asformat('csc')
            L0_I_norm = phi.T.dot(L0_I.dot(phi))
        
            #overlapping factor
            f = L0_I_norm/L0_in_norm

            #complex wavenumber
            k = mode[0]-1.j*mode[1]
            
            #gamma factor
            gamma = self.pump_params['gamma_perp'] / ( k - self.pump_params['k_a'] + 1.j * self.pump_params['gamma_perp'])

            #shift in k            
            k_shift = k*np.sqrt( (1. + gamma*f*D0_0) / (1. + gamma*f*D0_1) ) - k

            return k_shift
        
    def pump_trajectories(self, modes, params, D0_max, D0_steps):
            """
            For a sequence of D0, find the mode positions, of the modes modes. 
            """

            self.D0s = np.linspace(0., D0_max, D0_steps) #sequence of D0
            new_modes = [modes.copy(), ] #to collect trajectory of modes 
            self.pump_params['D0'] = self.D0s[0]

            for iD0 in tqdm(range(D0_steps-1)):
                #print('D0:', self.D0s[iD0+1], )

                for m in range(len(modes)):
                    #estimate the shift in k
                    k_shift = self.pump_linear(new_modes[-1][m], self.D0s[iD0], self.D0s[iD0+1])
                    
                    #shift the mode to estimated position
                    new_modes_init = new_modes[-1].copy()
                    new_modes_init[m,0] += np.real(k_shift)
                    new_modes_init[m,1] -= np.imag(k_shift)
                    
                #set the pump to next step and correct the mode
                self.pump_params['D0'] = self.D0s[iD0+1]
                new_modes.append(np.array(self.update_modes(new_modes_init, params)))

            return np.array(new_modes) 

    def plot_pump_traj(self, Ks, Alphas, s, modes, new_modes, estimate = False):
        self.plot_scan(Ks,Alphas,s, modes)

        if modes is not None:
            plt.plot(modes[:,0], modes[:,1],'ro')

        for i in range(len(modes)):
            D_th = self.linear_lasing_threshold(modes[i], self.D0s[0])
            
            #if D_th < self.D0s[-1] and D_th>0:
            #    plt.plot(new_modes[:,i,0],new_modes[:,i,1],'r-.')
            #else:
            plt.plot(new_modes[:,i,0],new_modes[:,i,1],'k-', lw=2)

        #plt.plot(new_modes[-1,:,0],new_modes[-1,:,1],'k+')

        ax = plt.gca()
        for i in range(len(modes)):
            dx = new_modes[-1,i,0]-new_modes[-2,i,0]
            dy = new_modes[-1,i,1]-new_modes[-2,i,1]

            #plt.arrow(new_modes[-1,i,0], new_modes[-1,i,1], dx, dy, head_width=0.005, head_length=0.01, fc='k', ec='k')
            ax.annotate("", xy=(new_modes[-1,i,0], new_modes[-1,i,1]), xytext=(new_modes[-2,i,0], new_modes[-2,i,1]), arrowprops=dict(facecolor='black', shrink=0.05))


        if estimate:
            for m in range(len(modes)):

                for iD0 in range(len(self.D0s)-1):
                    D_th = self.linear_lasing_threshold(new_modes[iD0][m], self.D0s[iD0])

                    k_shift = self.pump_linear(new_modes[iD0][m], self.D0s[iD0], self.D0s[iD0+1])
                    
                    if D_th < self.D0s[-1]-self.D0s[iD0] and D_th>0:
                        plt.scatter(new_modes[iD0][m][0]+np.real(k_shift), new_modes[iD0][m][1]- np.imag(k_shift), s=30, c='r')
                    else:
                        plt.scatter(new_modes[iD0][m][0]+np.real(k_shift), new_modes[iD0][m][1]- np.imag(k_shift), s=30, c='k')

                    plt.plot([new_modes[iD0][m][0], new_modes[iD0][m][0]+np.real(k_shift)], [new_modes[iD0][m][1], new_modes[iD0][m][1] - np.imag(k_shift)], c='k', lw = 0.8)
                    
                    
                 
    def full_lasing_threshold(self, modes, params, tol, D0_max, D0_steps):
            """
            For a sequence of D0, find the mode positions, of the modes modes. 
            """
            
            self.D0s = np.linspace(0., D0_max, D0_steps) #sequence of D0s we'll use later

            lasing_threshold_single_modef = partial(self.lasing_threshold_single_mode, params, tol, D0_max, D0_steps)
            with Pool(processes = self.n_processes_scan) as p_th:  #initialise the parallel computation
                out = list(tqdm(p_th.imap(lasing_threshold_single_modef, modes), total = len(modes))) #run them 

            K_ths = []
            D0_ths = []
            
            #loop over all modes
            for m in range(len(modes)):
                #save the treshold lasing mode
                K_ths.append(out[m][0])
                D0_ths.append(out[m][1])
                
            return K_ths, D0_ths
                
    def lasing_threshold_single_mode(self, params, tol, D0_max, D0_steps, mode):

                #first see if the linear approximation has a lasing threshold under the max
                D0_th = self.linear_lasing_threshold( mode, 0)
                if D0_th < D0_max and D0_th>0: #if it is, run the fine search (positive for modes going downwards)
                    k_th, D0_th = self.search_threshold( mode, params, tol, D0=0, D0_step_max = D0_steps, D0_max = D0_max)
                    
                    if D0_th > D0_max: #if it is larger than max, set it to not lasing
                        k_th = -1
                        D0_th = -1
                        
                else: #if it is not increase D0 step by step until it does or not

                    new_modes = [mode, ] #to collect trajectory of modes 
                    D0_th = 0
                    for iD0 in range(D0_steps-1):
                        D0_th = self.D0s[iD0] + self.linear_lasing_threshold( new_modes[-1], self.D0s[iD0])
                        
                        k_shift = self.pump_linear(new_modes[-1], self.D0s[iD0], self.D0s[iD0+1])
                        
                        #shift the mode to estimated position
                        new_mode_init = new_modes[-1].copy()
                        new_mode_init[0] += np.real(k_shift)
                        new_mode_init[1] -= np.imag(k_shift)
                        
                        self.pump_params['D0'] = self.D0s[iD0+1]
                        params['reduc'] = 0.5
                        
                        k_new = self.find_mode_brownian_ratchet(new_mode_init, params, disp = False, save_traj = False)

                        #check if it has been found
                        if len(k_new)>1:
                            new_modes.append(k_new) 
                        
                        #if not, try until the end of times
                        else: 
                            print('run many attempts')
                            att = 0 
                            while len(k_new)==1:
                                att +=1
                                k_new = self.find_mode_brownian_ratchet(new_mode_init, params, disp = False, save_traj = False)
                            new_modes.append(k_new) #compute the real wavenumber
                
                            print(att, 'attempts to find a mode, think of fine tuning parameters! (linear increments)')
                
                        
                        if D0_th < D0_max and D0_th>0: #if it is, run the fine search (positive for modes going downwards)
                            k_th, D0_th = self.search_threshold( new_modes[-1], params, tol, D0 = self.D0s[iD0+1], D0_step_max = D0_steps, D0_max = D0_max)
                            
                        else: #if the mode will not lase, set to -1
                            k_th = -1
                            D0_th = -1
                            
                return k_th, D0_th
        
        
    def search_threshold(self, mode_init, params, tol, D0, D0_step_max, D0_max):
    
        s_size_0 = params['s_size'].copy() #remember the original step size (to be reduced in the search later on)
        self.pump_params['D0'] = D0
        new_modes = [mode_init, ] #to collect trajectory of modes 
        D0_th_previous = D0 #estimate the D_threshold from D0
        k_imag = new_modes[-1][1]
                
        D0_th = self.linear_lasing_threshold(new_modes[-1], D0_th_previous)
        D0_th_orig = D0_th.copy()
        while abs(k_imag) > tol:
            
            #estimate the increment in D0 to get to D0_th
            D0_step = self.linear_lasing_threshold(new_modes[-1], D0_th_previous) 
            
            #if larger that the max allowed step size, set the step to max 
            #(this prevents long jumps and getting a different mode with the search)
            if D0_step > D0_step_max:
                D0_step = D0_step_max
                
            D0_th =  D0_th_previous + D0_step
            
            #if we get a threshold too large, set it to the mid-point instead (this avoids oscillations to blow up)
            if D0_th > D0_max: 
                D0_th = 0.5*(D0_th_previous + D0_th)
                
            k_shift = self.pump_linear(new_modes[-1],  D0_th_previous,  D0_th)
            #shift the mode to estimated position
            new_mode_init = new_modes[-1].copy()
            new_mode_init[0] += np.real(k_shift)
            new_mode_init[1] -= np.imag(k_shift)
            
            #rescale the step sizes when we get closer to the solution
            params['s_size']    = (1e-3 + 1e-1*abs((D0_th -  D0_th_previous)/D0_th))*s_size_0 

            self.pump_params['D0'] = D0_th #set the estimated pump
            k_new = self.find_mode_brownian_ratchet(new_mode_init, params, disp = False, save_traj = False) #find the new mode
            
            #check if it has been found
            if len(k_new)>1:
                new_modes.append(k_new) 
                
            else: #if not, try until the end of times
                att = 0 
                print('Start running many attempts (threshold search)')
                while len(k_new)==1:
                    att +=1
                    k_new = self.find_mode_brownian_ratchet(new_mode_init, params, disp = False, save_traj = False)
                new_modes.append(k_new) #compute the real wavenumber
                
                print(att, 'attempts to find a mode, think of fine tuning parameters! (threshold search)')
                
            D0_th_previous = D0_th.copy()
            k_imag = new_modes[-1][1]
            
            params['s_size'] = s_size_0 #set the step size back to normal
        
        return new_modes[-1], D0_th   
    
    
    def linear_lasing_threshold(self, mode, D0):
            """
            Estimate the lasing threshold using linear approximation, for a mode with pumping D0
            """
                    
            #update the laplacian
            self.pump_params['D0'] = D0
            self.update_chi(mode)
            self.update_laplacian()
            
            
            #compute the node field
            phi = self.compute_solution()
            
            self.Z_matrix_U1() #compute the Z matrix
            edge_norm = self.Winv.dot(self.Z).dot(self.Winv) #compute the correct weight matrix
            
            #compute the inner sum
            L0_in = self.BT.dot(edge_norm.dot(self.in_mask)).dot(self.B).asformat('csc')
            L0_in_norm = phi.T.dot(L0_in.dot(phi))
            
            #compute the field on the pump
            L0_I = self.BT.dot(edge_norm.dot(self.pump_mask.dot(self.in_mask))).dot(self.B).asformat('csc')
            L0_I_norm = phi.T.dot(L0_I.dot(phi))
        
            #overlapping factor
            f = L0_I_norm/L0_in_norm

            #complex wavenumber
            k = mode[0]-1.j*mode[1]
            
            #gamma factor
            gamma = self.pump_params['gamma_perp'] / ( k - self.pump_params['k_a'] + 1.j * self.pump_params['gamma_perp'])
            
            #Q-value
            Q = mode[0]/(2*mode[1])
            #estimated D_th
            D_th = 1./(Q*np.imag(-gamma)*np.real(f))

            return D_th
