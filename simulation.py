"""
    Creating a sumulation: Simulation class
    ------------------------------------------------------------------
    
    Reference:
        Gillespie, D.T. (1977) Exact Stochastic Simulation of Coupled Chemical Reactions.
        J Phys Chem, 81:25, 2340-2361.
        
    Code written by:
        Manhyuk Han, 2021 <manhyukhan@kaist.ac.kr>
        Korea Advanced Institute of Science and Technology
        Created: 2021-12-06

    preview:

    sim = Simulation()
    sim.addPropensity(propensity_func,*args)
    sim.addReaction(reaction_matrix)
    sim.addInit(x0=x0)
    sim.run(
        type = 'steadystate', ## 'timeevolution'
        rep = 1000,
        tspan = [tini, tfinal],
        nthread = 20
    )

    sim.run(
        type = 'steadystate',
        rep = 1000,
        tspan = [tini, tfinal],
        nthread = 20
    )
    
    X = sim.getState()

    sim.addInit(x0=X[-1,:])     # addInit method initializes the initial value
    sim.addPropensity(propensity_func,*args)    # add method automatically initializes
    
    sim.run(
        type = 'timeevolution',
        rep = 1000,
        tspan = [tini, tfinal],
        nthread = 20
    )
    
    X = sim.getState()
"""
import numpy as np
import numba
from contextlib import closing
import sys
import os
import time
import logging
import warnings
import multiprocessing as mp

def init(*args):

    global propensityFunction__
    global reactionMatrix__
    global tspan__
    global MAX_OUTPUT_LENGTH__
    propensityFunction__ = args[0]
    reactionMatrix__ = args[1]
    tspan__ = args[2]
    MAX_OUTPUT_LENGTH__ = args[3]
    

class Simulation():
    
    def __init__(self):
        """Declare attirbutes of the systems.
        
        numStates = number of the state, int
        time = column vector of timepoint of the simulation, np.ndarray
        timeEvolve = equipartition time vector, np.ndarray
        state = rank 2 tensor of time x numStates, np.ndarray
        tspan = [tinitial, tfinal], list
        thread = number of thread (multiprocessing),int
        propensityFunction = propensity function, mapping state-like vector to reaction-like vector, callable
        reactionMatrix = rank 2 tensor of numReaction x numStates
        """
        self.numStates = int()
        self.time = np.empty(0)
        self.timeEvolve = np.empty(0)
        self.state = np.empty((0,0))
        self.tspan = [0,0]
        self.propensityFunction = lambda x: np.transpose(x)
        self.reactionMatrix = np.empty((0,0))
       
    def addInit(self,x0):
        """add Initial state (n-dimensional row vector) on the system.
        this method automatically initializes state. To restart the simulation, you need to run with restart=True
        """        
        if isinstance(x0,list):
            x0 = np.array(x0)
        
        self.numStates = len(x0)
        self.state = x0.reshape(1,self.numStates)
        
    def addPropensity(self, propensity_func, *args):
        """add propensity function (callable) on the system.

        Args:
            propensity_func (callable): generate propensity vector (numReaction x 1)
        
        """
        self.propensityFunction = lambda x: propensity_func(x, *args)
           
    def addReaction(self, reaction_matrix):
        """add reaction matrix (stochi_matrix) on the system.

        Args:
            reaction_matrix (rank 2 tensor: numReaction x numStates)
        """
        self.reactionMatrix = reaction_matrix
    
    def getState(self):
        """get state matrix
        """
        return self.state
    
    def getTimeEvolve(self):
        """get timeEvolve vector
        """
        return self.timeEvolve
    
    @staticmethod
    @numba.njit
    def _iterator(
                initialState,
                propensityFunction=propensityFunction__,
                reactionMatrix=reactionMatrix__,
                tspan=tspan__,
                MAX_OUTPUT_LENGTH=MAX_OUTPUT_LENGTH__):
        """Iteration unit
        
        type/ plausibility check must be preceded
        """
        ## running simulation
        numStates = len(initialState)
        timeVec = np.zeros(MAX_OUTPUT_LENGTH,1)
        stateTen = np.zeros(MAX_OUTPUT_LENGTH,numStates)
        timeVec[0] = tspan[0]
        stateTen[0,:] = initialState
        rxnCount = 0
        
        # MAIN LOOP
        while timeVec[rxnCount] < tspan[1]:
            # calculate propensity function
            a = propensityFunction(stateTen[rxnCount,:])
            
            a0 = sum(a)
            r = np.random.rand(1,2)
            tau = -np.log(r[0,0]) / a0

            mu = 0
            s = a[0]
            r0 = r[0,1] * a0
            while s < r0:
                mu += 1
                s += a[mu]
            
            if rxnCount + 1 > MAX_OUTPUT_LENGTH:
                timeVec = timeVec[:rxnCount,1]
                stateTen = stateTen[:rxnCount,:]
                warnings.warn('Number of reaction events exceeded the number pre-allocated. Simulation is terminated')
                return timeVec, stateTen

            timeVec[rxnCount+1,1] = timeVec[rxnCount,1] + tau
            stateTen[rxnCount+1,:] = stateTen[rxnCount,:] + reactionMatrix[mu,:]
            rxnCount += 1
        
        # remove padding
        timeVec = timeVec[:rxnCount+1,1]
        stateTen = stateTen[:rxnCount+1,:]
        assert stateTen.shape == (rxnCount+1,numStates)
        assert timeVec.shape == (rxnCount+1,1) 
        
        if timeVec[-1] > tspan[1]:
            timeVec = timeVec[:-1]
            stateTen = stateTen[:-1,:]
        
        return [timeVec, stateTen]
    
    @staticmethod
    @numba.njit
    def _timeaverage(results,
                     tspan,
                     numStates):
        """
        """
        timeEvolve = np.linspace(tspan[0],tspan[1]-1,tspan[1])
        state = np.zeros((tspan[1],numStates),dtpye=np.float32)

        for result in results:
            T = result[0]
            X = result[1]
        
            binnedT = np.bincount(T)
            timeEvolve += binnedT
            ind = 0
            
            for i in range(tspan[1]):
                for j in range(binnedT[i]):
                    state[i,:] += X[j+ind,:]
                ind += binnedT[i]
            
            assert ind == len(T)-1
            
        state /= len(results)
            
        return timeEvolve, state
    
    
    def run(self,
            runType = 'steadystate',
            rep = 100,
            tspan = [0,2500],
            nproc = 4,
            MAX_OUTPUT_LENGTH = 1000000,
            restart = False
            ):
        """Running Gilespie simulation

        Args:
            type (str, optional): type of simulation. Defaults to 'steadystate'.
            rep (int, optional): repetition of simulation. Defaults to 100.
            tspan (list, optional): timespan over which simulation performed. Defaults to [0,2500].
            nthread (int, optional): number of threads. Defaults to 8.
        """
        self.tspan = tspan
        self.nproc = nproc
        self.numReaction = self.reactionMatrix.shape[0]
        
        # assertions and warnings
        assert len(self.state[0,:]) == self.numStates
        assert self.numStates == self.reactionMatrix.shape[1]
        
        if not (type(tspan[0]) == int and type(tspan[1]) ==int):
            warnings.warn('timespan must be interger. they will be automatically floored')
            self.tspan[0] = int(tspan[0])
            self.tspan[1] = int(tspan[1])
        
        if (not restart) and (self.state.shape == (self.numStates,)):
            raise ValueError('state must be initialized if not restarting the simulation!')
        try:
            temp = self.propensityFunction(self.state[0,:])
            assert len(temp) == self.numReaction
        except:
            raise ValueError('propensity function is malfunctioning')
        
        ## multithreading
        #values = [filenames[i::n] for i in range(n)]]
        initialState = self.state[-1,:]
        inValues = [np.concatenate([initialState for j in range(rep)])]
        assert len(inValues) == rep
        assert len(inValues[0]) == self.numStates
        
        init(self.propensityFunction,self.reactionMatrix,self.tspan,MAX_OUTPUT_LENGTH)
        if __name__=="__main__":
            with closing(mp.Pool(processes=nproc)) as p:
                results = p.map(self._iterator,inValues)
            assert len(results)==rep and type(results[0])==list
        
        ## steadystate
        if not runType.lower() in ['steadystate','timeevolution']:
            warnings.warn('runType not clear. Assume it steadystate')
            runType = 'steadystate'
        
        if runType.lower() == 'steadystate':
            self.state = np.concatenate([result[1][-1,:].reshape(1,self.numStates) for result in results], axis=0)
        elif runType.lower() == 'timeevolution':
            self.timeEvolve, self.state = self._timeaverage(results,self.tspan,self.numStates)
            
            
            
    
                
                
                
            
        
        
        
        
        

    
