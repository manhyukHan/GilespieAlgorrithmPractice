# GilespieAlgorrithmPractice
This practice is for implementing gilespie algorithm in python.

Goal:
    Practice OOP for gilespie algorithm implement
    class Simulation: multiprocessing, gilesipe algorithm

    simulate histone modification
    looping/ linear spreading

    Contact probability
     References
     F. Erdel, K. Müller-Ott and K. Rippe, Ann. N. Y. Acad. Sci., 2013, 1305, 29–43.
     K. Rippe, Trends Biochem. Sci., 2001, 26, 733–740.
    
    Simulation object
     add reaction matrix (stochiometric matrix)
     add propensity function (function of n-dimensional Real, and *arg)
     add arguments for propensity function
     add timespan
     add initial state

     add # of processors
     add # of threads

     run simulation
     ## save simulation result  --- for gilespie algorithm, saving each simulation may be non-significant!
        - binary (h5 format)
        - dat format
     
     steady state (convergence)

    preview:

    sim = Simulation()
    sim.addPropensity(propensity_func,*args)
    sim.addReaction(reaction_matrix)
    sim.addInit(x0=x0)
    sim.run(
        type = 'steadystate', ## 'timeevolution'
        reporter = 'dat',     ## 'h5', False (if reporter is False, it returns result like regular function)
        rep = 1000,
        tspan = [tini, tfinal],
        nthread = 20
    )
    """
    X = sim.run(
            type = 'steadystate',
            reporter = False,
            rep = 1000,
            tspan = [tini, tfinal],
            nthread =20
        )
    """
    sim.addInit(x0=X[-1,:])     # addInit method initializes the initial value
    sim.addPropensity(propensity_func,*args)    # add method automatically initializes
    sim.run(...)






