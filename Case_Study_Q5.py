#In Q5 parameters v can change from period to period and then in period k price S can take 2^k values. To properly test calibration functions
#I also write code to price European call options when parameters v change. In addition, I allow interest rate r to be any number between 0 and 1
#and calculate risk-neutral probabilities. These probabilities are going to be different for each period too, and it is easier to store all
#the information in a binary tree.

import numpy as np
import math
import timeit
import scipy.special


#Node will store two variables: one for a price S of an asset (or value of a European call option or anything else) and the other for probability of
#some event. Also a node has three pointers: two for two children nodes and one for one parent node.
#I can link nodes by using methods add_parent() and add_nodes(). And I can print two variables in the node.

class Node: #nodes for Q5.
    def __init__(self,value,prob):
        self.value=value #can store here some price, or value of option at some period of time
        self.prob=prob #can store here probability of some event.
        
        self.down=[] #these are to connect nodes
        self.up=[]
        self.par=[]
    
    def add_parent(self,parent_Node):
        self.par=parent_Node
        
    def add_nodes(self,new_node_down,new_node_up):
        self.down=new_node_down
        self.up=new_node_up
        
    def print_node(self):
        print("Node has ",self.value,"and probability is",self.prob)
        

#I also introduce class Tree. A tree is just a collection of linked nodes. The main method I use is list_of_leaves() that returns all the leaves
#(nodes without children).
  
class Tree:
    def __init__(self,root):
        self.root=root
    
    def print_tree(self):
        self.root.print_node()
        if self.root.down!=[]:
            left_new_root=self.root.down
            Left_Tree=Tree(left_new_root)
            Left_Tree.print_tree()
            right_new_root=self.root.up
            Right_Tree=Tree(right_new_root)
            Right_Tree.print_tree()
        
    def list_of_leaves(self): #will often need to obtain the leaves of a tree
        if self.root.down==[]:
            return [self.root]
        else:
            left_new_root=self.root.down
            Left_Tree=Tree(left_new_root)
            left_leaves=Left_Tree.list_of_leaves()
            right_new_root=self.root.up
            Right_Tree=Tree(right_new_root)
            right_leaves=Right_Tree.list_of_leaves()
            return left_leaves+right_leaves


        
# Function value() calculates the value of the option for the given price S of the asset and strike price K
def value(S,K):
    return max(S-K,0)



#Function European_pricing() calculates the value of the European call option. I create a Node with value of the price S=1 in period 0 and probability is 1.
#I start a Tree with the root in this node.
#Then for each period I obtain the list of leaves of the tree. In period k there is 2^k leaves and each leaf has a possible value of price S in it with the
#corresponding probability. Knowing all possible values of the price S in period k I calculate all (2^(k+1)) possible values of the price S in period k+1 with
#corresponding probabilities. By period N the Tree has all possible paths for price movements.
#Next I calculate the values of the European call option in period N - this information is in leaves.
#Then using the same formula as in Q1 I 'climb' the tree to the root calculating values of the option for each period and deleting leaves after using them
#for computations.
#The price of the European option will be in the root.

def European_pricing(N,K,V,p,r,S0):
    
    #we need to make the tree for price paths
    root=Node(S0,1) #store here the price at period 0. S_0=1 with probability 1
    Option_Pricing_Tree=Tree(root) #grow the tree with the root

    for i in range(0,N): #repeat the following procedure N periods:
    
        current_leaves=Option_Pricing_Tree.list_of_leaves() #obtain the list of leaves of the tree.
        #They have possible prices in the
        
        for node in current_leaves: #for each leaf (where I keep possible prices in the current period)
            current_price=node.value #I take the value of the price stored in
        
            up_move=current_price*(1+V[i]) #calculate two possible values of the prices for the next period
            down_move=current_price*(1-V[i])
        
            new_node_1=Node(up_move,p[i])  #create two more nodes to store these two new prices for next period
            new_node_2=Node(down_move,1-p[i])
        
            node.add_nodes(new_node_2,new_node_1) #connect these nodes to the tree
            new_node_1.add_parent(node)
            new_node_2.add_parent(node) #these nodes are leaves now
    

#Now the paths for prices of the asset are generated and stored in the tree.
#I calculate the value of the option at the date of maturity. The prices of the asset at period N are in the leaves of the tree

    current_leaves=Option_Pricing_Tree.list_of_leaves() # I get the list of leaves with the prices
    for node in current_leaves:
        temp_value=node.value
        node.value=value(temp_value,K) #calculate the values of option at period N and store them in the leaves


    while Option_Pricing_Tree.root.down!=[]: #climb the tree to the root: until I reach it, I do the following
    
        current_leaves=Option_Pricing_Tree.list_of_leaves() #obtain all the leaves (with the values of the option at the latest period)
        for node in current_leaves: #for each leaf I obtain the parent of the leaf
            parent_node=node.par 
            if parent_node.down!=[]:  #calculate the value of the option at previous period for the given parental node
                parent_node.value=(parent_node.down.prob*parent_node.down.value+parent_node.up.prob*parent_node.up.value)/(1+r)
                parent_node.down=[] #delete the children and all the relevant information will be in childless leaf
                parent_node.up=[]
    
#At the end there will be only root left with the value of the option in it.
    return Option_Pricing_Tree.root.value



#Function risk_neutral_p() calculates the risk-neutral probability for given parameter v and interest rate r
def risk_neutral_p(r,v):
    return 0.5+r/(2*v)



#Function difference() is used to calibrate values v. Namely, for the given value of parameter v I calculate corresponding risk neutral probability.
#Then I obtain list of leaves of the Calibration tree that stores all the values v from the previous periods I calibrated already.
#I can then obtain the period k, and it means there is k-1 values v_1,v_2,...,v_(k-1) are calibrated.
#As I know v_1,...v_(k-1), I can calculate possible prices S in period k-1 and all these prices (with corresponding probabilities) are stored in the
#leaves. Then for the given v I calculate value of the European call option, provided I know previous v_1,v_2,...,v_(k-1). And finaly I return the 
#difference between this value and the target value of some European call option with date of maturity k. If this difference is approximately 0,
#it means that current value v is next calibrated v_k.

def difference(v,Calibration_Tree,K,V0,r):
    S=0
    p=risk_neutral_p(r,v)
    current_leaves=Calibration_Tree.list_of_leaves() #get all the possible prices in the current period
    k=math.log(len(current_leaves),2)+1 # this is the period
    for node in current_leaves:
        S=S+value(node.value*(1+v),K)*node.prob*p+value(node.value*(1-v),K)*node.prob*(1-p) #calculate here expected value
    return S/(1+r)**k-V0

#Function binary_section() is used for calibration of v_1,v_2,v_3,...v_N. Let's assume I have calibrated v_1,v2,...,v_(k-1) already and the prices S are in
#Calibration Tree. Then I have to find v_k knowing the value of some European call option with maturity date k. The value of this option
#can be represented as a function of v. After that, the idea is the same as in Q2.


def binary_section(l,r,Calibration_Tree,K,X,rate):
    m=(l+r)/2 # middle of the interval of [l,r]
    diff_l=difference(l,Calibration_Tree,K,X,rate) #find the values of the auxilary function at the ends
    diff_m=difference(m,Calibration_Tree,K,X,rate) #of the given interval and in the middle
    diff_r=difference(r,Calibration_Tree,K,X,rate)
    
    c=rate+0.001 #It might be that the solution v is not uniqie (similarly to Q2). Namely, if small v (r+0.001) solves the optimisation problem, then
    if abs(diff_l)<0.00001 and l==c: #there is infinitely many v that solve the same problem. In this case I send a warning and stop calibration.
        print("Warning! Cannot calibrate v precisely as it is too small.")
        return -1
   #recursively shrink the interval until the value of the function at some end of the interval is aproximately 0 and root is detected
    if abs(diff_l)>0.0001 or abs(diff_r)>0.0001:
        if diff_l*diff_m<0: #If at the left end and in the middle the values of the function have different signs, the root is in the (l,m)
            m=binary_section(l,m,Calibration_Tree,K,X,rate) 
        else:
            m=binary_section(m,r,Calibration_Tree,K,X,rate) # otherwise the root is in (m,r)
    
    return m



#Function calibration() finds a unique vector  v_1,v_2,...,v_N (if such a vector exists). I create a node (S0,1) - price S of the asset at time 0 is
#equal to 1 with probability 1. This node will be the root of calibration tree.
#There is a vector of option prices where each option has different date of maturity. For each period k I do the following:
#1. pick the next value of the European option with maturity k.
#2. use brute force to find the value of the parameter v_k (similarly to Q2) knowing already calibrated v_1,v_2,..,v_(k-1).
#3. check if the value v_k is not too small to avoid not precise and not unique solutions.
#4. update the calibration tree by storing possible prices S in period k in the new leaves.

def calibration(K,option_prices,r,S0):
    root=Node(S0,1) #S_0=1 with probability 1
    Calibration_Tree=Tree(root) #start growing the tree from this root
    v_vector=[] # will keep calibrated v1,v2,...v_N in this list
    
    while len(option_prices)>0:
        X=option_prices[0] #take the values of the options one by one and delete it from the list
        option_prices.pop(0) #this way the first element in the list will be the price of the next option to consider
        
        v=r+0.001 #start with very small v>r
        S=0
        current_leaves=Calibration_Tree.list_of_leaves() #get all the possible prices in the current period
        k=math.log(len(current_leaves),2)+1
        while abs(S-X)>0.00001: #check, if v gives approximation of V0 after plugged into the formula for the value of the European option
            S=0
            p=risk_neutral_p(r,v)
            for node in current_leaves:
                S=S+value(node.value*(1+v),K)*node.prob*p+value(node.value*(1-v),K)*node.prob*(1-p)
            S=S/(1+r)**k
            v=v+0.00001
        c=r+0.00101
        if v==c:
            print("Warning! Cannot calibrate v precisely as it is probably too small")
            return -1
    
        v_vector.append(v) # found v and keep it in the vector
        p=risk_neutral_p(r,v)
        #update the tree, knowing some values of v.
        for node in current_leaves:
            price_moves_down=node.value*(1-v) #knowing v1,v2,..,v_k I can calculate possible values for S_k+1
            new_prob_down=node.prob*(1-p) #and corresponding probabilities
            price_moves_up=node.value*(1+v)
            new_prob_up=node.prob*p
            
            new_node_down=Node(price_moves_down,new_prob_down) #make nodes with values for S_k+1 and connect them to the tree
            new_node_up=Node(price_moves_up,new_prob_up)
            node.add_nodes(new_node_down,new_node_up)
            new_node_up.add_parent(node)
            new_node_down.add_parent(node)
    return v_vector

  
#Function calibration_fast() uses the same ideas, as function calibration(), but instead of brute force search it uses binary section (similar to Q2).

def calibration_fast(K,option_prices,S0,r):
    
    root=Node(S0,1) #S_0=1 with probability 1
    Calibration_Tree=Tree(root) #start growing the tree from this root
    v_vector=[] # will keep calibrated v1,v2,...v_N in this list
    while len(option_prices)>0:
        X=option_prices[0] #take the values of the options one by one and delete it from the list
        option_prices.pop(0) #this way the first element in the list will be the price of the next option to consider
        
        
        v=binary_section(r+0.001,0.9999,Calibration_Tree,K,X,r) #v is the root of the auxilary monotonically
        #increasing function, and can be found by binomial section.
        if v==-1:
            return -1
        
        v_vector.append(v) # found v and keep it in the vector
        
        #update the tree, knowing some values of v.
        current_prob=risk_neutral_p(r,v)
        current_leaves=Calibration_Tree.list_of_leaves()
        for node in current_leaves:
            price_moves_down=node.value*(1-v) #knowing v1,v2,..,v_k I can calculate possible values for S_k+1
            new_prob_down=node.prob*(1-current_prob) #and corresponding probabilities
            price_moves_up=node.value*(1+v)
            new_prob_up=node.prob*current_prob
            
            new_node_down=Node(price_moves_down,new_prob_down) #make nodes with values for S_k+1 and connect them to the tree
            new_node_up=Node(price_moves_up,new_prob_up)
            node.add_nodes(new_node_down,new_node_up)
            new_node_up.add_parent(node)
            new_node_down.add_parent(node)
    return v_vector


#Functions check_for_pricing() and check_for_calibration() just check if parameters of the model are valid. For instance, strike price K has to be
#positive

def check_for_pricing(K,r,V,p,S0):
    if(S0<0):
        print("Error: price of the asset must be positive")
        return 1
    if(K<0):
        print("Error: strike price must be positive")
        return 1
    if(len(V)!=N) or (len(p)!=N):
        print("Error: the size of vector V or p doesn't match the number of periods")
    if(r<0) or (r>1):
        print("Error: interest rate has to be a number from (0,1)")
        return 1
    for i in range(0,N):
        if(V[i]>1) or (V[i]<0) or (p[i]>1) or (p[i]<0):
            print("Error: either v or p in some period is not from (0,1)")
            return 1
    return 0

def check_for_calibration(option_prices,K,r,S0): # here just check if inputs is correct
    if (K<0):
        print("Error: strike price must be positive")
        return 1
    if r<0 or r>1:
        print("Error: interest rate must be in (0,1) for calibration")
        return 1
    if S0<0:
        print("Error: price of the asset has to be positive")
        return 1
    return 0



#Next I test how everything is working. Before calibration I want to price some European call options and to do this, I generate
#v1,v2,...,v_N (and calculate risk-neutral probabilities).


K=1.1 #K is a strike price,
S0=1 #S_0 is the price of asset at period 0;
r=0.0 #interest rate is r=0, but if we want to price European call option, it can be any number 0<r<1
N=10 # number of periods;
V_main=np.random.uniform(r,1,N) #vector of v_j, for the price changes: S_j+1=S_j(1+v_j) or S_j+1=S_j(1-v_j).
#V_main has uniformly distributed v_1,v_2,...,v_N.
p_main=[]
for i in range(0,N):
    p_main.append(0.5+r/(2*V_main[i])) #here I define risk-neutral probability, if r=0.0, then p=0.5.

option_prices_1=[]
option_prices_2=[]
for i in range(0,N): #in this loop I price N European call options with different dates of maturity with parameters defined above.
    V=V_main[:i+1]
    p=p_main[:i+1]
    c=check_for_pricing(K,r,V,p,S0)
    if(c==0):
        n=len(V)
        Price=European_pricing(n,K,V,p,r,S0)
        option_prices_1.append(Price)
        option_prices_2.append(Price) #option_prices will store the prices of the European options
        print("Price of the European call option is",Price," date of maturity is",i+1)
    else:
        print("Not correct input")
        
#Finally, I test calibration on the given option_prices
print("Given prices for ",N," options are", option_prices_1)

v_vector=calibration_fast(K,option_prices_2,S0,r)
if v_vector!=-1:
    print("Calibrated values v are (binary section)",v_vector)

v_vector=calibration(K,option_prices_1,r,S0)
if v_vector!=-1:
    print("Calibrated values v are (brute force)",v_vector)
