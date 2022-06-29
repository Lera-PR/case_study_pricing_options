import numpy as np
import math
import timeit
import scipy.special
import matplotlib.pyplot as plt



# Function value() calculates the value of the option for the given price S of the asset and strike price K
def value(S,K):
    return max(S-K,0)


#Function risk_neutral_p() calculates the risk-neutral probability for given parameter v and interest rate r
def risk_neutral_p(r,v):
    return 0.5+r/(2*v)

#Function F() is used to calibrate the value of parameter v for Q2. It calculates the value of European call option for given v and
# compares it to the given target value of the European call option. If it returns a small number, approximately equal to 0, then
#the required value of the parameter v is detected.
def F(v,N,K,V0,r):
    S=0
    p=risk_neutral_p(r,v)
    for i in range(0,N+1):
        S=S+scipy.special.comb(N,i)*value((1+v)**i*(1-v)**(N-i),K)*p**i*(1-p)**(N-i)
    return S/(1+r)**N-V0



#Function binary_section() is also used to calibrate the parameter v for Q2. Namely, value of a European call option can be represented as a monoton
#increasing function of v on the interval (r,1) where r is interest rate. Binary section is looking for the root of this function. It starts with the
#interval (r,1) and calculates the values of the function at r(+0.001), 0.9999 and (r+1)/2. Then it compares the signs of the values of the function
#at r and (r+1)/2. If the signs are different, then the root is in (r,(1+r)/2). Otherwise the root is in ((1+r)/2,1). We repeat till the root is detected.

#There is a possible problem with calibration of the parameter v. It turns out that in some cases we cannot give a precise answer. Consider the following
#case: N=2, K=0.5, v=0.1, r=0, and p=0.5. If I price European call option with such parameters I get V0=0.5.
# If I change parameter v=0.2 and other parameters are the same, then the value of the European option is still V0=0.5.
#Hence, if I have to calibrate v knowing that V0=0.5, K=0.5 and N=2 and r=0.0, I will not be able to give a single number, only an interval for
#possible values of parameter v.
#The other possible issue is that there is no such a v that would give targte V0 and no solution at all
# Then it means that in the binary section my intervals will be very close to 1 and I have to check for it as well. Otherwise there is recurssion error.

def binary_section(l,r,N,K,X,rate): #this is function for the fast calibration in Q2
    if(l>=0.9998): #check if solution v exists in this loop
        print("Warning! No v for such a value of the European call option is found. Solution doesn't exist")
        return -1
    
    m=(l+r)/2 # middle of the interval of [l,r]
    diff_l=F(l,N,K,X,rate) #find the values of the auxilary function at the ends
    diff_m=F(m,N,K,X,rate) #of the given interval and in the middle
    diff_r=F(r,N,K,X,rate)
    
    c=rate+0.001 #Here I check if very little (r+0.001) value of v solves the calibration problem. If it does, the answer to Q2 is not uniqie.
    if abs(diff_l)<0.00001 and l==c:
        print("Warning! Cannot calibrate v precisely as it is too little. Solution is not uniqie")
        return -1
    
   #recursively shrink the interval until the value of the function at some end of the interval is aproximately 0 and root is detected
    if abs(diff_l)>0.00001 or abs(diff_r)>0.00001:
        if diff_l*diff_m<0: #If at the left end and in the middle the values of the function have different signs, the root is
            m=binary_section(l,m,N,K,X,rate) #in the [l,m]
        else:
            m=binary_section(m,r,N,K,X,rate) # otherwise the root is in [m,r]
    
    return m


#solution to Q4 is in expectation_max_price()
#I look for the expected value of the maximum of price S using Monte-Carlo simulation. Namely, I generate a path for prices S of the underlying
#asset knowing how exactly price S might change from period to period. I go along the path and store the maximum price S I came across.
#I repeat this process of price path generation I=2000000 times and obtain the list of 2000000 prices that were maximums for each path.
# By the Law of Large Numbers, average of the sample approximates the mathematical expectation of maximal price S.

def expectation_max_price(v,p,S0,N):
    I=2000000 #number of Monte Carlo simulation
    max_S=np.ones(I) #this will be a sample of maxS for each or I simulations.
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



#solution to Q1 and Q3 is in pricing_option()
#First, I generate all possible pathes of the price S starting from period 0 at to period N and store these paths in the matrix S.
#I use the fact that parameter v and corresponding p do not change in this model and it means that at period k price S can take
#only k+1 values (not 2^k). The exact order of price movements before period k is not important, only the number of "up"s matters.
#This fact saves computational time and memory.
#Hence, the element S[i][j] in the matrix is the price S in period j and price S moved up j-i times before.
#After all the paths are generated, I calculate the value of the option at period N using value() function. This step is the same for both European and American
#call options.
#Then I use the standard formulas for European and American call option prices. Namely, if I know the values of a European call option at date k+1,
#then the value at date k and price S is V_k(S)=(pV_(k+1)(S(1+v))+(1-p)V_(k+1)(S(1-v)))/(1+r). Where p is risk-neutral price and r is an interest rate.
#For American option the formula is slightly different: V_k(S)=max((pV_(k+1)(S(1+v))+(1-p)V_(k+1)(S(1-v)))/(1+r),max(S-K,0))
#because American call option might be executed at any date.
#Using these formulas, I calculate the values of the options in periods N-1, N-2 and so on and the answer is in S[0][0]


def pricing_option(v,p,S0,N,o,K,r):
    S=np.zeros((N+1,N+1),dtype=np.float) #need to generate all price paths first. This matrix will be used to store them
    S[0][0]=S0
    for i in range(1,N+1): 
        S[0][i]=S[0][i-1]*(1+v) #first row will contain path where price S moves only up.
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
                S[i][j]=max((p*S[i][j+1]+(1-p)*S[i+1][j+1])/(1+r),value(S[i][j],K)) ##standard discounted expectation formula, just
                #added the comparison to max(S-K,0) for American option.
                
    else:
        print("Don't recognise the type of option")
        return -1
    return S[0][0]



#Two versions of the solution to Q2 are in functions calibration() and calibration_fast().
#Function calibration() uses brute force search to find the value of the parameter v.
#It starts with a small possible value for v, which is (r+0.001) and calculate the value of a European call option with such a v as a parameter.
#If it mathes the target price V0 of some European call options, then v is a solution to calibration problem.
#If it doesn't match V0, then we change v slightly and check again.
#Calibration problem doesn't always have a unique solution v: it might have infinite number of solutions, or no solution at all
    
def calibration(K,N,V0,r): #looking for the proper v using brute force
    v=r+0.001 #start with very small v>r
    S=0
    while abs(S-V0)>0.00001: #check, if v gives approximation of V0 after plugged into the formula for the value of the European option
        if(v>=0.9999): #if the candidate for solution v is approaching 1, it means there is (very likely) no solution.
            print("Warning! No v for such a value of the European call option is found. Solution doesn't exist")
            return -1
        S=0
        p=risk_neutral_p(r,v)
        for i in range(0,N+1):
            S=S+scipy.special.comb(N,i)*value((1+v)**i*(1-v)**(N-i),K)*p**i*(1-p)**(N-i) #here is the value of the European option with the given v.
        S=S/(1+r)**N
        v=v+0.00001
    c=r+0.00101 #Calibration problem doesn't always have a unique solution v, and I check in this loop, if there is potential issue. Namely,
    if v==c:# if a small number v solves the calibration problem, then there is a interval for other possible values v that will solve the problem too.
        print("Warning! Cannot calibrate v precisely as it is probably too little")
        return -1
    return v



#Function calibration_fast() uses binary section to find a root of a monoton increasing function. I start with the interval (r,1) for values v, and
#using binary section of the interval check if the required v is in (r,(1+r)/2) or ((1+r)/2,1) and then repeat for a new shorter interval.
def calibration_fast(K,N,V0,r):
    v=binary_section(r+0.001,0.9999,N,K,V0,r)
    return v



# Function check_of_parameters() simply checks if the values of the parameters in the model are valid. For instance, value v has to be larger then
#interest rate r. 
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
        
        

        
#Here I test how everything is working

v=0.1 #the value that defines movments of the asset price
r=0.0 #interest rate; it is 0 in the model, but I can take any 0<r<1 and adjust risk-neutral probability measure in the code
p=0.5+r/(2*v) #risk-neutral probability of price of the asset moving up
S0=1 #price of the asset at period 0
N=2 #number of periods
K=0.6 # strike price
o='E' # variable that can be 'E' or 'A' depending on the type of the option


E_max_S=expectation_max_price(v,p,S0,N) #calculate the expected value of maxS
print("Mathematical expectation of max of the prices is",E_max_S)

V_E=pricing_option(v,p,S0,N,o,K,r) #pricing of European call option
print("value of the European call option is:",V_E)

o='A'
V_A=pricing_option(v,p,S0,N,o,K,r) #pricing of American call option
print("value of the American call option is:",V_A)

#V_E =? I can either enter some number, or use the obtained one to test
est_v=calibration(K,N,V_E,r)
if(est_v>-1):
    print("Parameter v calibrated using brute force",est_v)

est_v=calibration_fast(K,N,V_E,r) #Knowing the price of European call option, number of periods, probabilities of moves
if(est_v>-1):
    print("Parameter v calibrated using binary section",est_v)


