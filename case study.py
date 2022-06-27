import numpy as np
import math
import timeit
import scipy.special
import matplotlib.pyplot as plt


def value(S,K):
    return max(S-K,0) #simply calculate the value of the option provived some value of the price

def F(v,N,K,p,V0,r): #monotonically increasing auxiliary function I use for calibration in Q2
    S=0
    for i in range(0,N+1):
        S=S+scipy.special.comb(N,i)*value((1+v)**i*(1-v)**(N-i),K)*p**i*(1-p)**(N-i)
    S=S/(1+r)**N
    return S-V0

def binary_section(l,r,N,K,current_prob,X,int_rate): #this is function for the fast calibration in Q2
    m=(l+r)/2 # middle of the interval of [l,r]
    diff_l=F(l,N,K,current_prob,X,int_rate) #find the values of the auxilary function at the ends
    diff_m=F(m,N,K,current_prob,X,int_rate) #of the given interval and in the middle
    diff_r=F(r,N,K,current_prob,X,int_rate)
    
   #recursively shrink the interval until the value of the function at some end of the interval is aproximately 0 and root is detected
    if abs(diff_l)>0.00001 and abs(diff_r)>0.00001:
        if diff_l*diff_m<0: #If at the left end and in the middle the values of the function have different signs, the root is
            m=binary_section(l,m,N,K,current_prob,X,int_rate) #in the [l,m]
        else:
            m=binary_section(m,r,N,K,current_prob,X,int_rate) # otherwise the root is in [m,r]
    
    return m


#solution to Q4
def expectation_max_price(v,p,S0,N):
    I=50 #number of Monte Carlo simulation
    max_S=np.ones(I) #this will be a sample of maxS for each or I simulations.
    for i in range(0,I): #repeat the following I times
        path=np.random.binomial(1,p,N)
        price_moves=1+v*(-1)**path #generate N random variables, that will represent up or down price moves
        S=S0
        for j in range(0,N): #calculate the values of the price in each period.
            S=S*price_moves[j]
            if(max_S[i]<S): #if current maximum price we saw is less than the price in this period i, change the maximum
                max_S[i]=S
    print(max_S)
    #After repeating this I times, we have sample of I realisations of maxS.
    E_max_S=np.sum(max_S)/I #expectation will be approximated by the average of the sample
    return E_max_S



#solution to Q1 and Q3
def pricing_option(v,p,S0,N,o,K,r):
    S=np.zeros((N+1,N+1),dtype=np.float) #need to generate all price paths first. This matrix will be used to store them
    S[0][0]=S0
    for i in range(1,N+1): 
        S[0][i]=S[0][i-1]*(1+v) #first row will contain path where price moves only up.
        for j in range(1,N+1): #column by column generate possible prices for each period
            S[j][i]=S[j-1][i-1]*(1-v)

    for i in range(0,N+1):
        S[i][N]=value(S[i][N],K) # the values of the option on the expiration date
        
    if o=='E': #pricing European option
        for j in range(N-1,-1,-1):
            for i in range(N-1,-1,-1):
                S[i][j]=(p*S[i][j+1]+(1-p)*S[i+1][j+1])/(1+r) #standard discounted expectation formula. S[i][j+1] is the value 
                #or the option at period j+1 provided price moved up from period j (with probability p) and S[i+1][j+1] is the 
                #value of the option at period j+1 provided price moved down from period j (with probability p)
    
    elif o=='A': #pricing American option
        for j in range(N-1,-1,-1):
            for i in range(N-1,-1,-1):
                S[i][j]=max((p*S[i][j+1]+(1-p)*S[i+1][j+1])/1+r,value(S[i][j],K)) ##standard discounted expectation formula, just
                #added the comparison to max(S-K,0) for American option.
                
    else:
        print("Don't recognise the type of option")
        return -1
    return S[0][0]



#Two versions of the solution to Q2
def calibration(K,N,V0,p,r): #looking for the proper v using brute force
    if V0==0:
        #all pathes lead to price < strike price, hence (1+v)^N<K, hence, v<K^(1/N)-1
        print("v is less than",K**(1/N)-1)
    else:
        v=0.0 #start with very small v
        S=0
        while abs(S-V0)>0.001: #check, if v gives approximation of V0 after plugged into the formula for the value of the European option
            S=0
            for i in range(0,N+1):
                S=S+scipy.special.comb(N,i)*value((1+v)**i*(1-v)**(N-i),K)*p**i*(1-p)**(N-i)
            S=S/(1+r)**N #here is the value of the European option with the given v.
            v=v+0.001
        return v


def calibration_fast(K,N,V0,p,r): # Looking for the root of a monotonically increasing function 
    #of the strictly increasing function by using binary section of the interval (0,1)
    if V0==0:
        #all pathes lead to price < strike price, hence (1+v)^N<K, hence, v<K^(1/N)-1
        print("v is less than",K**(1/N)-1)
    else:
        v=binary_section(0.0001,0.9999,N,K,p,V0,r)
        return v

        
        

#Here I test how my functions work
v=0.3 #the value that defines movments of the asset price
p=0.5 #probability of price of the asset moving up
S0=1 #price of the asset at period 0
N=4 #number of periods
K=1.1 # strike price
o='E' # variable that can be 'E' or 'A' depending on the type of the option
r=0 #interest rate; it is 0 in the model, but can be any number 0<r<1


E_max_S=expectation_max_price(v,p,S0,N) #calculate the expected value of maxS
print("Mathematical expectation of max of the prices is",E_max_S)

V_E=pricing_option(v,p,S0,N,o,K,r) #pricing of European call option
print("value of the European call option is:",V_E)

o='A'
V_A=pricing_option(v,p,S0,N,o,K,r) #pricing of American call option
print("value of the American call option is:",V_A)

#est_v=calibration(K,N,V_E,p,r)
est_v=calibration_fast(K,N,V_E,p,r) #Knowing the price of European call option, number of periods, probabilities of moves
print(est_v) #interest rate and the strike price we calibrate value v.
