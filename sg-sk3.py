#this is an attempted implementation of the arXiv paper https://arxiv.org/pdf/1806.08815.pdf

# The general idea is that you want to find

# want to find how many MC steps will get you to a good estimate of
# the scaling, with the idea that if you go long enough in time(MC steps), you'll
# find the reference energy for any given run, but you'd like to
# see how short you can go in time (MC steps)

# therefore, as a function of time, see what the distributions are like
# with respect to getting the minimum energy

# SK model, 1D
# inspired from: Lewis Cole's blog: https://lewiscoleblog.com

import numpy as np
import matplotlib.pyplot as plt


# Pick random seed
# another way to initialize is to make a call to datetime, but you'll
# want to keep the initial see you use to be able to reproduce results
np.random.seed(17)

# Set size of model N 
N = 50

# print(f'spins {spins}') 

# number of timesteps 
timesteps = 2000

# interaction sets the interaction energy between neighbouring spins
# initialize it here, and it stays fixed
# note that this isn't periodic boundary conditions at the moment

interaction = np.zeros((N, N))
for i in range(N):
    for j in range(i):
        interaction[i, j] = np.random.randn() / np.sqrt(N)
        interaction[j, i] = interaction[i, j]


# beta=1/k_b T, really, beta=1/T
# system in glassy-phase for T<s so beta>1/s. 
# here, s=1. if s != 1, you should include it in the line of the interaction
# as : interaction[i, j] = np.random.randn()*s / np.sqrt(N)

T=1
beta = 1/(0.75*T)

# Define update step
dE = 0

def monte_carlo_spin_flip(s_array, i_array):
    """
    update function performs 1 update step to the model
    
    inputs:
    s_array - an array of N spins (+-1)
    i_array - an array of interaction strengths NxN
    """
    
    # Select a spin to update
    site = np.random.choice(s_array.shape[0], 1)[0]
    
    # Get interaction vector
    i_vector = i_array[site,:]
    
##    print(f'i_vector {i_vector.shape} i_array {i_array.shape} site {site}')
    # Calculate energy change associated with flipping site spin
    dE = 2*np.dot(i_vector, s_array)*s_array[site]
    
    # Calculate gibbs probability of flip
    prob = np.exp(-beta*dE)
    energynow = -1 * np.dot(s_array, np.dot(s_array, i_array)) / 2
    
#    print(f' energy {dE} prob {prob} energynow {energynow}')
    # Sample random number and update site
    if dE <= 0 or prob > np.random.random():
        s_array[site] *= -1
    else:
        dE = 0
    return s_array,dE

def main_loop(timesteps , s_array, i_array):
    s_temp = s_array.copy()
    energy = np.zeros(timesteps+1)
    energy[0] = -1 * np.dot(s_array, np.dot(s_array, i_array)) / 2
    for i in range(timesteps):
        update_step,dE = monte_carlo_spin_flip(s_temp, i_array)
        s_temp = update_step
        energy[i+1] = energy[i] + dE
#        energynow = -1 * np.dot(s_temp, np.dot(s_temp, interaction)) / 2
    return energy


def find_ref_energy(num_replicas):
    # spins set the initial configuration for the spins in a 1D N size matrix 
    # I'm going to try num_replicas different initial conditions, with the same
    # interaction matrix. Each one of these is called a replica
    spins = np.random.choice([-1, 1], N)
    # Calculate initial values
#    energy[0] = -1 * np.dot(spins, np.dot(spins, interaction)) / 2

    # this finds the energy as a function of time in each replica
#    energy_all_time_replicas=[]
    
    energy_all_time_replicas=[]
    energy_min_replica=[]
    for i_replica in range(num_replicas):
        spins = np.random.choice([-1, 1], N)

        energy_all_time_replicas.append(main_loop(timesteps*10, spins, interaction))
#        energy_all_time_replicas=main_loop(timesteps*10, spins, interaction)

    #find lowest energy in each replica
    for i_time in range(len(energy_all_time_replicas)):
        energy_min_replica.append(  min(energy_all_time_replicas[i_time]))
    print (f' energy min replica {energy_min_replica}')
    # global min
    # take as the reference energy everything within 5% of min
    energy_ref=min(energy_min_replica)*.95
    print (f' energy min {energy_ref} ')
    print (f' energy min {energy_ref} {energy_ref*.95} ')

#   count what fraction of replicas get to reference energy
    num_replicas_near_energy_ref = len([i for i in energy_min_replica if i < energy_ref])

    print(f'num replicas below ref e {num_replicas_near_energy_ref} of {num_replicas}')
#    print(f'min e repl {energy_min_replica}')

#### Run Main Loop
num_replicas=10
find_ref_energy(10)

# now to find posterior
# https://stats.stackexchange.com/questions/68803/how-to-calculate-the-bayesian-posterior-probability-from-observations


# plot energy evolving in time
# fig, ax1 = plt.subplots()
# ax1.set_xlabel("Time step")

# ax1.set_ylabel("Energy")
# ax1.plot(energy)

# plt.show()

