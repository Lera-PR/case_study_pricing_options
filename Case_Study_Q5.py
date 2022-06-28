#!/usr/bin/env python
# coding: utf-8

# In[544]:


import numpy as np
import math
import timeit
import scipy.special

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


# In[545]:


def value(S,K):
    return max(S-K,0)

#in this function I calculate the value of the European call option
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
                parent_node.down=[] #delete the children
                parent_node.up=[]
    
#At the end there will be only root left with the value of the option in it.
    return Option_Pricing_Tree.root.value


# In[546]:


def risk_neutral_p(r,v):
    return 0.5+r/(2*v)


def difference(v,Calibration_Tree,K,V0,r): #axiliary function that I use for calibration v
    #the auxilary function is monotonicaly increasing on [0,1] with respect to v
    S=0
    p=risk_neutral_p(r,v)
    current_leaves=Calibration_Tree.list_of_leaves() #get all the possible prices in the current period
    k=math.log(len(current_leaves),2)+1 # this is the period
    for node in current_leaves:
        S=S+value(node.value*(1+v),K)*node.prob*p+value(node.value*(1-v),K)*node.prob*(1-p) #calculate here expected value
    return S/(1+r)**k-V0

def binary_section(l,r,Calibration_Tree,K,X,rate):
    m=(l+r)/2 # middle of the interval of [l,r]
    diff_l=difference(l,Calibration_Tree,K,X,rate) #find the values of the auxilary function at the ends
    diff_m=difference(m,Calibration_Tree,K,X,rate) #of the given interval and in the middle
    diff_r=difference(r,Calibration_Tree,K,X,rate)
    
    c=rate+0.001
    if abs(diff_l)<0.00001 and l==c:
        print("Warning! Cannot calibrate v precisely as it is too little.")
        return -1
   #recursively shrink the interval until the value of the function at some end of the interval is aproximately 0 and root is detected
    if abs(diff_l)>0.0001 or abs(diff_r)>0.0001:
        if diff_l*diff_m<0: #If at the left end and in the middle the values of the function have different signs, the root is
            m=binary_section(l,m,Calibration_Tree,K,X,rate) #in the [l,m]
        else:
            m=binary_section(m,r,Calibration_Tree,K,X,rate) # otherwise the root is in [m,r]
    
    return m

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
            print("Warning! Cannot calibrate v precisely as it is probably too little")
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
    if len(V)!=len(p):
        print("Error: vector of v and vector of p have different sizes")
        return 1
    for i in range(0,N):
        if(V[i]>1) or (V[i]<0) or (p[i]>1) or (p[i]<0):
            print("Error: either v or p in some period is not from (0,1)")
            return 1
    return 0

def check_for_calibration(option_prices,K,p,r,S0): # here just check if inputs is correct
    n=len(option_prices)
    if(n!=len(p)):
        print("Error: sizes of vectors of option prices and probabilities do not match")
        return 1
    
    for i in range(0,n): #this loop is to make sure that the calibrated values are precise.
        if p[i]>1 or p[i]<0: 
            print("Error: probabilities have to be in (0,1)")
            return 1
    if (K<0):
        print("Error: strike price must be positive")
        return 1
    if r!=0:
        print("Error: interest rate must be 0 for calibration")
        return 1
    if S0<0:
        print("Error: price of the asset has to be positive")
        return 1
    
    return 0


# In[570]:


K=0.5 #K is a strike price,
S0=1 #S_0 is the price of asset at period 0;
r=0.2 #interest rate is r=0, but if we want to price European call option, it can be another number 0<r<1
#N=10 # number of periods;
#V=np.random.uniform(r,1,N) #vector of v_j, for the price changes: S_j+1=S_j(1+v_j) or S_j+1=S_j(1-v_j)
#print(V)
V=[0.59780795]
print(V)
N=len(V)
p=[]
for i in range(0,N):
    p.append(0.5+r/(2*V[i])) #here we define risk-neutral probability, if r=0.0, then p=0.5.


#will price European call option given known variables above:
c=check_for_pricing(K,r,V,p,S0)
if(c==0):
    Price=European_pricing(N,K,V,p,r,S0)
    print("Price of the European call option is",Price)
else:
    print("Not correct input")


# In[572]:


#here will calibrate vector of v knowing the prices of N European call options.
N=10
option_prices=[0.61045239674046,0.6708646383855166,0.7652845129241485,0.804860218326258,0.837499545944606,0.8597186202729517,
               0.8962482476073704,0.9219968588658805,0.9392911995919495,0.9483384015302669]
print("Given prices for ",N," options are", option_prices)

v_vector=calibration(K,option_prices,r,S0)
if v_vector!=-1:
    print("Calibrated values v are",v_vector)
option_prices=[0.61045239674046,0.6708646383855166,0.7652845129241485,0.804860218326258,0.837499545944606,0.8597186202729517,
               0.8962482476073704,0.9219968588658805,0.9392911995919495,0.9483384015302669]
v_vector=calibration_fast(K,option_prices,S0,r)
if v_vector!=-1:
    print("Calibrated values v are",v_vector)
#else:
#    print("Not correct input")


# In[ ]:





# In[ ]:




