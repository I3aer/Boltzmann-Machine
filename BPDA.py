import numpy as np


class BMJPDA(object):
    """
    Boltzmann machine to solve the data association problem. BM is a symetrically interconnected network
    of neuron-like units. Those units make stochastic decisions whether to be on or off . This means that 
    the Boltzmann machine accepts or rejects the elemental hypotheses represented by their units at each 
    iteration. The aim of the Boltzmann machine is to find probable configuration of units by minimizing 
    an energy function. Thus, it will have a stationary distribution which is equal to the distribution 
    over valid joint association events. To this end, the Boltzmann machine uses the simulated annealing 
    schedule. The Boltzmann machines are stacked to form a parallel computational organization. This leads 
    to the computationally efficient and accurate solutions to constraint satisfaction tasks. 
    """

    def __init__(self, distortion_matrix):
        
        # This is the distance matrix defined between measurements and targets
        self.distortion_matrix = distortion_matrix
        
        self.meas_number = distortion_matrix.shape[0]
        self.target_number = distortion_matrix.shape[1]
        
        self.nodes_number = self.meas_number * self.target_number
        
        # The temperature at thermal eqilibrium
        self.equilibrium_temp = 0.1
        
        # The coefficients of the inhibition terms in the energy function
        self.A = 10 #one neuron is on per row
        self.B = 10 #one neuron is on per column

        # Grid of nodes represents binary neurons. Each row represents a measurement and each column represents a target. 
        self.nodes = np.zeros(shape=[self.meas_number,self.target_number])
        
        # Initial configuration of the BM is set to represent that all targets are missed.
        self.nodes[0,:] = 1
        
        self.temperature = self._get_initial_temperature()
        
    def _get_initial_temperature(self):
        """
        Initial temperature should be so high that the the positive energy transitions will
        be accepted with a high probability. Thus, the network can climb out of a local minima.
        """
        return 2*self.A*(self.target_number-1)+2*self.B*(self.meas_number-2)+self.B

    def update(self):

        # Get random coordinates for a node from a discrete uniform distribution 
        i = np.random.random_integers(low=0, high=self.meas_number-1)
        j = np.random.random_integers(low=0, high=self.target_number-1)

        change_probability = self.get_activation_change_probability(i,j)

        # Change node activation with change_probability
        if (np.random.binomial(1, change_probability) == 1):

            # Change node activation
            self.nodes[i, j] = 1 - self.nodes[i, j]

            if self.is_nodes_configuration_legal():
                self.last_legal_configuration = self.nodes.copy()
    
    def anealling_shedule(self):
        '''An exponential temperature annealing schedule'''
        self.temperature *= 0.95
        
        # Stop the search if temperature reaches the equilibrium
        if (self.temperature > self.equilibrium_temp):
            return True
        else:
            return False
        
    def get_energy_change(self, i, j):
        """Compute the energy change due to change in the state of the randomly picked neuron"""

        node_value = self.nodes[i, j]
        #  if the target j is missed
        if (i==0):
            sum_row_nodes = 0
        else:
            sum_row_nodes = np.sum(self.nodes[i,:],axis=0) - node_value
            
        sum_col_nodes = np.sum(self.nodes[:,j],axis=0) - node_value
        
        distance = self.distortion_matrix[i, j]

        return (1-2*node_value)*(distance + self.B/self.temperature + 
                                 2*self.A/self.temperature*sum_row_nodes + 
                                 2*self.B/self.temperature*(sum_col_nodes-1)
                                )

    def get_activation_change_probability(self, i, j):
        """activation/deactivation probability follows a sigmoidal distribution""" 
        
        energy_change = self.get_energy_change(i, j)

        exponential_argument = energy_change/self.temperature

        return self.sigmoidal(exponential_argument)
    
    def is_nodes_configuration_legal(self):
        """
        check if BM configuration corresponds to a valid joint data association.
        """
    
        for row_index in xrange(1,self.nodes.shape[0]):
    
            row_sum = np.sum(self.nodes[row_index, :])
            if ( row_sum != 0 and row_sum != 1):
                return False
    
        for column_index in xrange(self.nodes.shape[1]):
            
            # column sum must be equal to 1 since dummy measurement
            if (np.sum(self.nodes[:, column_index]) != 1):
                return False

        # configuration is legal
        return True
    
    def get_last_legal_configuration(self):
        return self.last_legal_configuration

    def sigmoidal(self,e):
        if (e <= 0):
            return 1. / (1. + np.exp(e))
        elif (e > 0):
            return np.exp(-e) / (1 + np.exp(-e))
        
if __name__ == '__main__':
    
    # number of parallel BMs
    Nbm = 20
    # measurements including two clutters
    M = [6.5, 7, -1, 11, 2.5]
    # target states
    T = [5, 8, 0]
    # detection Probability 
    Pd = 0.98
    # measurement innovation variance
    S = 2.0   
    # number of trails in each annealing schedule
    Ntr = ((len(M)+1)*len(T))**2
    # global states of BMs
    global_states = []
    # (M+1)xN distance matrix
    D = np.zeros(shape=[len(M)+1,len(T)])
    # distortion is the minus log likelihood
    llk = lambda x,y,S: 0.5*np.log(2*np.pi*S) - np.log(Pd) + 0.5*(x-y)**2/S 
    
    for z in xrange(len(M)+1):
        if (z == 0):
            D[z,:] = -np.log(1 - Pd)
        else:
            D[z,:] = [llk(t,M[z-1],S) for t in T]
    
    for m in xrange(Nbm):
        tr = 0
        bm = BMJPDA(D)
        while(True):
            tr += 1
            print("BM: {0:d}, trial: {1:d}".format(m,tr))
            bm.update()
            if (tr % Ntr == 0):
                if not(bm.anealling_shedule()):
                    global_states.append(bm.get_last_legal_configuration())
                    break
                
    assoc_matrix = np.mean(global_states,axis=0)
    
    np.set_printoptions(precision=3, suppress=True)
    print("Association probabilities: {:s}".format(assoc_matrix))
                                