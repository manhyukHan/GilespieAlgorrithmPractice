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
    sim.addPropensity(propensity_func,**kwargs)    # add method automatically initializes
    
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
import dill
import multiprocessing as mp

propensityKwargs__ = dict()
propensityFunction__ = lambda x: x
reactionMatrix__ = np.empty((0,0))
tspan__ = [0,0]
MAX_OUTPUT_LENGTH__ = 0


def init(*args):

    global propensityFunction__
    global propensityKwargs__
    global reactionMatrix__
    global tspan__
    global MAX_OUTPUT_LENGTH__
    propensityFunction__ = args[0]
    reactionMatrix__ = args[1]
    tspan__ = args[2]
    MAX_OUTPUT_LENGTH__ = args[3]
    propensityKwargs__ = args[4]
    

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
        init(self.propensityFunction,self.reactionMatrix,self.tspan,0,dict())
       
    def addInit(self,x0):
        """add Initial state (n-dimensional row vector) on the system.
        this method automatically initializes state. To restart the simulation, you need to run with restart=True
        """        
        if isinstance(x0,list):
            x0 = np.array(x0)
        
        self.numStates = np.size(x0)
        self.state = np.copy(x0).astype(np.float32).reshape(1,self.numStates)
        
    def addPropensity(self, propensity_func, **kwargs):
        """add propensity function (callable) on the system.

        Args:
            propensity_func (callable): generate propensity vector (numReaction x 1)
        
        """
        global propensityKwargs__
        propensityKwargs__ = kwargs
        
        global propensityFunction__
        propensityFunction__ = propensity_func
        
        self.propensityFunction = lambda x: propensity_func(x, **kwargs)
        
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
    def _iterator(
                x,
                ):
        """Iteration unit
        
        type/ plausibility check must be preceded
        """
        global propensityFunction__
        global reactionMatrix__
        global tspan__
        global MAX_OUTPUT_LENGTH__
        global propensityKwargs__
        
        #print(propensityFunction__)
        
        propensityFunction = propensityFunction__
        propensityKwargs = propensityKwargs__
        reactionMatrix = reactionMatrix__
        tspan = tspan__
        MAX_OUTPUT_LENGTH = MAX_OUTPUT_LENGTH__
        
        ## running simulation
        initialState = np.copy(x).astype(np.float32).reshape(1,np.size(x))
        numStates = np.size(initialState)
        timeVec = np.zeros((MAX_OUTPUT_LENGTH,1),dtype=np.float32)
        stateTen = np.concatenate([initialState,np.zeros((MAX_OUTPUT_LENGTH-1,numStates),dtype=np.float32)],axis=0)
        timeVec[0,0] = tspan[0]
        rxnCount = 0
        
        @numba.njit
        def __action(a,rxnCount,timeVec,stateTen):
            a0 = np.sum(a)
            r = np.random.rand(1,2)
            tau = np.divide(-np.log(r[0,0]), a0)

            A = np.copy(a).astype(np.float32)
            mu = 0
            s = A[0]
            r0 = r[0,1] * a0
            while s < r0:
                mu += 1
                s += A[mu]

            timeVec[rxnCount+1,0] = np.add(timeVec[rxnCount,0],tau)
            stateTen[rxnCount+1,:] = np.add(stateTen[rxnCount,:],reactionMatrix[mu,:])
        
        # MAIN LOOP
        while timeVec[rxnCount] < tspan[1]:
            # calculate propensity function
            a = propensityFunction(stateTen[rxnCount,:],**propensityKwargs)
            __action(a,rxnCount,timeVec,stateTen)
            
            if rxnCount + 1 > MAX_OUTPUT_LENGTH:
                timeVec = timeVec[:rxnCount,0]
                stateTen = stateTen[:rxnCount,:]
                warnings.warn('Number of reaction events exceeded the number pre-allocated. Simulation is terminated')
                return [timeVec, stateTen]
            rxnCount += 1
        
        # remove padding
        if timeVec[rxnCount,0] > tspan[1]:
            timeVec[rxnCount,0] = tspan[1]
            #stateTen[rxnCount,:] = np.zeros(numStates).reshape(1,numStates)
        
        return [timeVec[:rxnCount+1,0], stateTen[:rxnCount+1,:]]
    
    @staticmethod
    def _timeaverage(t,     # np.ndarray in py.array
                     x,     # np.ndarray in py.array
                     Bins,  # np.ndarray
                     timeEvolve,    #np.ndarray
                     state, # np.ndarray, 2D, float32
                     rep):

        @numba.njit
        def __action(T,X,Bins,timeEvolve,state):
            binnedT, _ = np.histogram(T,bins=Bins)
            timeEvolve = np.add(binnedT, timeEvolve)
            ind = 0
            binnedT = np.copy(binnedT).astype(np.int64)
            
            for st in Bins[:-1]:
                for j in range(binnedT[st]):
                    state[st] += X[j+ind]
                if binnedT[st]: state[st] /= binnedT[st]
                ind += binnedT[st]
                
        for i in range(rep):
            T = np.copy(t[i]).astype(np.float32)
            X = np.copy(x[i]).astype(np.float32)
            __action(T,X,Bins,timeEvolve,state)           
            
        state = np.divide(state,rep)
            
        return timeEvolve, state
    
    
    def run(self,
            runType = 'steadystate',
            rep = 100,
            tspan = [0,2500],
            nproc = 4,
            MAX_OUTPUT_LENGTH = 1000000,
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
        assert np.size(self.state[0,:]) == self.numStates
        assert self.numStates == self.reactionMatrix.shape[1]
        
        if not (type(tspan[0]) == int and type(tspan[1]) ==int):
            warnings.warn('timespan must be interger. they will be automatically floored')
            self.tspan[0] = int(tspan[0])
            self.tspan[1] = int(tspan[1])
        
        if (self.state.shape != (1,self.numStates)):
            raise ValueError('state must be initialized if not restarting the simulation!')
        
        try:
            temp = propensityFunction__(self.state,**propensityKwargs__)
            assert len(temp) == self.numReaction
        except:
            raise ValueError('propensity function is malfunctioning')
        
        ## multithreading
        initialState = self.state
        assert (initialState==self.state).all()
        inValues = list()
        for j in range(rep):
            inValues.append(initialState)
        assert (inValues[0] == initialState).all()
        assert np.size(inValues[0]) == self.numStates
        
        argsets = [propensityFunction__,self.reactionMatrix,self.tspan,MAX_OUTPUT_LENGTH,propensityKwargs__]
        if nproc > 1:
            with closing(mp.Pool(processes=nproc, initializer=init, initargs=argsets)) as p:
                results = p.map(self._iterator,inValues)
            #print(results[-1][0])
        else:
            init(*argsets)
            results = list()
            proceed = 1
            for value in inValues:
                results.append(self._iterator(value))
                print(f'Simulation proceeding--------({proceed}/{rep})-')
                proceed+=1
               # print(results[-1][0])
        time.sleep(0.5)
        assert len(results)==rep
        
        t = list()
        x = list()
        for i in range(rep):
            t.append(results[i][0])
            x.append(results[i][1])
        
        ## steadystate
        if not runType.lower() in ['steadystate','timeevolution']:
            warnings.warn('runType not clear. Assume it steadystate')
            runType = 'steadystate'
        
        if runType.lower() == 'steadystate':
            #results = np.array(results,dtype=np.ndarray)
            self.state = np.concatenate([(1/10)*xx[-10:,:].sum(0) for xx in x],axis=0).reshape(rep,self.numStates)
            
        elif runType.lower() == 'timeevolution':
            Bins = np.linspace(self.tspan[0],self.tspan[1],self.tspan[1]+1,dtype=np.int64)
            #results = np.array(results,dtype=np.ndarray)
            timeEvolve = np.zeros(self.tspan[1],dtype=np.float32)
            state = np.zeros((self.tspan[1],self.numStates),dtype=np.float32)
            #t = np.array(results[:,0],dtype=np.ndarray)
            #x = np.array(results[:,1],dtype=np.ndarray)
            self.timeEvolve, self.state = self._timeaverage(t,x,Bins,timeEvolve,state,rep)
        
        print('_______Simulation Complete________')