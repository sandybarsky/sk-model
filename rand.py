
import numpy as np
import matplotlib.pyplot as plt
import random

'''
This program is an implementation of https://arxiv.org/pdf/1806.08815.pdf
section IV : meausring the time-to-solution(TTS) of a system with random variables.
Instead of spin glasses, I've used vectors (arrays) of random numbers. The 
reference energy solution is when one of the elements of the vector is 
greater than a threshold.

At the end a couple of graphs are shown: one for the quartiles of the 
posterior distribution for various vector lengths, the other graph is
the TTS as a function of vector length.
'''
# the tts is the key metric: it is the 99% prob=R99 that you get an energy in the desired
# range * a time to do a run. In this code, the time to do a "run" is a strictly linear
# function  of the length of the vecto

# define the class data. 
# Although this is a simple program in general the data gets poked at a lot
# and it's nice to have it in one spot


class data:
    n = 1
    a = 1
    b = 1
    def __init__(self, n,a,b,p,R99):
       self.n = n # "length" of random array sampled
       self.a = a #  a of the beta posterior
       self.b = b #  a of the beta posterior
       self.p=p  
       self.R99 = R99 #  data
    def mean(self):
       this_mean=np.mean(self.R99)
       return this_mean
    def tts(self):
       this_tts=self.mean()*n
       return this_tts
    def tts_quantile(self):
       quant=[]
       quant_of_data=[]
       for iq in range(10):
           quant.append((iq+1)/10)
           quant_of_data.append(np.quantile(self.R99*n,(iq+1)/10))
       return quant,quant_of_data


# I'm fixing the random seed. In general, it's good to have a random seed somehow
# either fixed or noted, so reults can be reproduced
np.random.seed(17)



#count how many samples > threshold

# Core parameters.
# a,b (alpha, beta) of prior distribution
# threshold is the "reference energy" value, so that if an element
# of a vector of random numbers has a value > threshold, that vector
# is said to have reached the reference energy. this is considered success


# rep is the number of vectors we are creating to find our distributions
# since rep is number of vectors we are creating to fix our distribution the
# value of rep has a huge impact on the value of R99, for small values of
# the vector length
# setting rep=100 is mostly a nice smooth curve for R99


a=0.5 # a, b (or alpha, beta) of the beta function for the prior
b=0.5 #Jeffries' parameters for prior
rep=100  # this is the number of runs to establish the posterior distribution

threshold=.90  
vector_length=[2,4,6,8,10,16,32,64]

# a few lists for easy plotting once we're done.
success_rate=[]
mean=[]
dist=[]
tts=[]
quant=[]
for n in vector_length:
    num_success=0
 
    
    for repeat in range(rep):
        ran_num=np.random.random(n)
        success = len([i for i in ran_num if i > threshold])
        if success >=1:
            num_success +=1
    success_rate.append(num_success)

    # posterior. We're pretending that we  have enough information
    # to accurately describe the posterior.
    beta_a=num_success + a     # a, b on the prior beta distribution
    beta_b=rep-num_success + b  # 0.5 = Jeffries'

    # now we have a distribution
    # let's find the R99

    p=np.random.beta(beta_a,beta_b,5000)
    R99=np.array(np.log(1-.99)/np.log(1-p))
    this_data=data(n,beta_a,beta_b,p,R99)
    this_mean=this_data.mean()
    this_tts=this_data.tts()
    this_quant= this_data.tts_quantile()
    dist.append( this_data)
    mean.append( this_mean)
    quant.append( this_quant[1])
    tts.append( this_tts)


# let's look at data

for iplt in range(len(vector_length)):

    plt.plot(this_quant[0],quant[iplt],label=(f'n={vector_length[iplt]}'))
plt.legend()
plt.title('quantiles of TTS for different lengths n')
plt.xlabel('quantiles')
plt.show()



fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.set_xlabel('N')
ax1.set_ylabel('R99', color='b')
ax2.set_ylabel('TTS: time-to-solution =R99* N', color='r')
ax1.plot(vector_length, mean,'bo-')
ax2.plot(vector_length, tts,'rx-')
plt.title(f'R99: average number of runs \n  to get a vector of length N \n to have one value above a reference value of {threshold}')

plt.show()



