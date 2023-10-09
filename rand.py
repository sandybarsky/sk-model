
import numpy as np
import matplotlib.pyplot as plt
import random


# define the class data. 
# Although this is a simple program in general the data gets poked at a lot
# and it's nice ot have it in one spot

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


# I'm fixing the random seed. In general, it's good to have a random seed somehow
# either fixed or noted, so reults can be reproduced
np.random.seed(17)



#count how many samples > threshold

# Core parameters. First we make rep number of vectors of lengths in
# vector length. Then we count how many of those vectors have one entry
# greater than threshold. If a vector has one entry greater than threshold
# it is considered a success

# we count the number of successes and together with a uniform flat prior
# distribution, we get a posterior (assumed beta function)

# since rep is number of vectors we are creating to fix our distribution the
# value of rep has a huge impact on the value of R99, for small values of
# the vector length
# setting rep=100 is mostly a nice smooth curve for R99


rep=100  # this is the number of runs to establish the posterior distribution

threshold=.9
vector_length=[2,3,4,5,6,7,8,10,16,20,26,32,50]

success_rate=[]
mean=[]
dist=[]
tts=[]
for n in vector_length:
    num_success=0
 
    
    for repeat in range(rep):
        ran_num=np.random.random(n)
        success = len([i for i in ran_num if i > threshold])
        if success >=1:
            num_success +=1
    success_rate.append(num_success)
    beta_a=num_success + 1     # the 1's on beta_a,b indicate that
    beta_b=rep-num_success +1  # the prior was flat

    # now we have a distribution
    # let's find the R99

    p=np.random.beta(beta_a,beta_b,5000)
    R99=np.array(np.log(1-.99)/np.log(1-p))
    this_data=data(n,beta_a,beta_b,p,R99)
    this_mean=this_data.mean()
    this_tts=this_data.tts()
    dist.append( this_data)
    mean.append( this_mean)
    tts.append( this_tts)



plt.plot(vector_length, mean,'bo-')
plt.plot(vector_length, tts,'rx-')
plt.xlabel('N')
plt.ylabel('R99 mean')
plt.title(f'R99: average number of runs \n  to get a vector of length N \n to have one value above a reference value of {threshold}')

#plt.title('R99: average number of runs \n  to get a vector of length N \n to have one value above a reference value of 0.9')
plt.show()



