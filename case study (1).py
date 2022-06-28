import numpy as np
import math
import timeit
import scipy.special
import matplotlib.pyplot as plt



def value(S,K):
    return max(S-K,0) #simply calculate the value of the option provived some value of the price S and strike price K

def risk_neutral_p(r,v): #calculate risk-neutral probability of price S moving up. If r=0.0, p=0.5, but I consider general case with 0<r<1
    return 0.5+r/(2*v)


#Function F(v,N,K,V0,r) is used to calibrate value v knowing the price of European call option (Q2).
#I simply calculate the value of European call option for a given v and compare to the target value V0.
#Value of European call oprion is a monotonically increasing function with respect to v.

def F(v,N,K,V0,r): #monotonically increasing auxiliary function I use for calibration in Q2
    S=0
    p=risk_neutral_p(r,v) #for given v I calculate risk-neutral probability
    for i in range(0,N+1):
        S=S+scipy.special.comb(N,i)*value((1+v)**i*(1-v)**(N-i),K)*p**i*(1-p)**(N-i) #After finishing the cycle S will be equal to the value of the Eur
    return S/(1+r)**N-V0

#Function binary_section() is looking for the single root of a monotonically increasing function on some interval.
#I start with interval (r,1) of possible values v, calculate the values of F() for the left and right edge of the interval and for the middle
#of the interval (r+1)/2. If signes of the value of the function F at r and at (1+r)/2 are different,
#then the root of the function is somewhere on the interval (r,(1+r)/2). Otherwise the root is on the interval ((1+r)/2,1). Repeat this process
#untill the value of F() is approximately 0 - this is where the root is.

def binary_section(l,r,N,K,X,rate): #"l" and "r" here are left and right bound of the interval, interest rate is "rate"
    m=(l+r)/2 # middle of the interval of (l,r)
    diff_l=F(l,N,K,X,rate) #find the values of the auxilary function at the ends
    diff_m=F(m,N,K,X,rate) #of the given interval and in the middle
    diff_r=F(r,N,K,X,rate)
    
   #recursively shrink the interval until the value of the function at some end of the interval is aproximately 0 and root is detected
    if abs(diff_l)>0.00001 or abs(diff_r)>0.00001:
        if diff_l*diff_m<0: #If at the left end and in the middle the values of the function have different signs, the root is
            m=binary_section(l,m,N,K,X,rate) #in the [l,m]
        else:
            m=binary_section(m,r,N,K,X,rate) # otherwise the root is in [m,r]
    return m


#function expectation_max_price() solves Q4. I use Monte-Carlo simulation method. Namely, I create a possible path from period 0 to N
#for a price S of the asset, knowing the risk-neutral probability of it moving up or down. While generating the path, I keep the maximum price S
#I came across along the path. I repeat this process I=2000000 times and end up with the array of 2000000 maximum prices from each of the pathes.
#By the Law of Large Numbers, average of this sample will be approximately equal to the mathematical expectation of maximal price.

def expectation_max_price(v,p,S0,N):
    I=2000000 #number of Monte Carlo simulation
    max_S=np.ones(I) #this will be a sample of maxS for each or I=2000000 simulations.
    for i in range(0,I): #repeat the following I times
        path=np.random.binomial(1,p,N)
        price_moves=1+v*(-1)**path #generate N random variables, that will represent up or down price moves
        S=S0
        for j in range(0,N): #calculate the values of the price in each period.
            S=S*price_moves[j]
            if(max_S[i]<S): #if current maximum price we saw is less than the price in this period i, change the maximum
                max_S[i]=S
    #After repeating this I times, we have sample of I realisations of maxS.
    E_max_S=np.sum(max_S)/I #expectation will be approximated by the average of the sample
    return E_max_S



#solution to Q1 and Q3
#function pricing_option() calculate the value of the European or American call option knowing parameter v, risk-neutral probability p,
#price of the asset at period 0, number of periods N, type of the option ('E' or 'A'), strike price K and interest rate r.
#I generate all the paths for the price S of the asset for N periods and store them in the matrix. It is convenient, because v and p are
#the same for each period and the order of 'ups' and 'downs' of the price S is not important, only the number of 'ups'. Hence, in period
#k price S can take k+1 possible values (not 2^(k+1)). So, in the given matrix, called S element in the row i and column j represents the
#price S of the asset in period j knowing that price moved up i times before (and j-i times it moved down).

#After all the prices path are generated, I calculate the value of the option at the date of maturite N. This is the same for both European and
#American call options.

#Then I go backwards using the standard formula for option pricing: if I know the values of the option at time k+1, I can calculate value of the option
#at time k (with price S) as follows: V_k(S)=(p*V_(k+1)(S(1+v))+(1-p)*V_(k+1)(S(1-v)))/(1+r)
#Repeating these calculations, I obtain the answer in S[0][0].

#For American call option the only difference is in the formula I apply to calculate V_k(S) knowing V_(k+1)(S(1+v)) and V_(k+1)(S(1-v)):
#Namely, V_k(S)=max((p*V_(k+1)(S(1+v))+(1-p)*V_(k+1)(S(1-v)))/(1+r),max(S-K,0))
#This comprison with max(S-K,0) is added as American option can be executed at any date, not necessarily at N.


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

#Function calibration() takes the strike price K, number of periods N, price of the European option V0 and interest rate, and return calibrated
#parameter v. It uses brute force: strating from r+0.001 (the minimal value of v possible in the model) it checks if the value of a European call option
# with such a v is close to the target V0. If not, then move to next possible value for v. I have to adjust risk-neutral probabilities for each v.
# We stop when for some v value of the European call option is approximately equal to the target V0.


def calibration(K,N,V0,r): #looking for the proper v using brute force
    if V0==0:
        #all pathes lead to price < strike price, hence (1+v)^N<K, hence, v<K^(1/N)-1
        print("v is less than",K**(1/N)-1)
    else:
        v=r+0.001 #start with very small v>r
        S=0
        while abs(S-V0)>0.00001: #check, if v gives approximation of V0 after plugged into the formula for the value of the European option
            S=0
            p=risk_neutral_p(r,v)
            for i in range(0,N+1):
                S=S+scipy.special.comb(N,i)*value((1+v)**i*(1-v)**(N-i),K)*p**i*(1-p)**(N-i) #here is the value of the European option with the given v.
            S=S/(1+r)**N
            v=v+0.00001
        return v


#Function calibration_fast() takes the same arguments as calibration(). The difference is that it looks for the root of a function on the
#interval (r+0.001,1). Namely, we can express price of a European call option as a function of v (risk-neutral probability is also functions of v and known
#interest rate r). Then, by using binary_section() we can quickly find the value v.
    
    
def calibration_fast(K,N,V0,r):
    if V0==0:
        #all pathes lead to price < strike price, hence (1+v)^N<K, hence, v<K^(1/N)-1
        print("v is less than",K**(1/N)-1)
    else:
        v=binary_section(r+0.001,0.9999,N,K,V0,r)
        return v
    
    
#Function check_of_parameters() simply checks if the values of parameters in the model are correct. For instance, interest rate cannot be negative
#and parameter v has to be larger then r.

def check_of_parameters(v,r,S0,K):
    if(v<0) or (v>1) or (v<r):
        print("Error: v is not a valid number")
        return -1
    if r<0 or r>1:
        print("Error: interest rate r is not a valid number")
        return -1
    if S0<1:
        print("Error: price of the asset has to be positive")
        return -1
    if K<0:
        print("Error: strike price has to be positive")
        return -1
    return 0

        
#In the next section I just test how pricing and calibration work.


v=0.4 #the value that defines movments of the asset price
r=0.1 #interest rate; it is 0 in the model, but I can take any 0<r<1 and adjust risk-neutral probability measure
p=0.5+r/(2*v) #risk-neutral probability of price of the asset moving up
S0=1 #price of the asset at period 0
N=4 #number of periods
K=1.1 # strike price
o='E' # variable that can be 'E' or 'A' depending on the type of the option

E_max_S=expectation_max_price(v,p,S0,N) #calculate the expected value of maxS
print("Mathematical expectation of max of the prices is",E_max_S)

V_E=pricing_option(v,p,S0,N,o,K,r) #pricing of European call option
print("value of the European call option is:",V_E)

o='A'
V_A=pricing_option(v,p,S0,N,o,K,r) #pricing of American call option
print("value of the American call option is:",V_A)


#Here I use V_E as the price of European call option calculated before 
est_v=calibration(K,N,V_E,r)
print("Parameter v calibrated using brute force",est_v)
est_v=calibration_fast(K,N,V_E,r)
print("Parameter v calibrated using the binary section",est_v) #interest rate and the strike price we calibrate value v.

