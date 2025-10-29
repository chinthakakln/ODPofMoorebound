

#This provides a python implementation of the 1st simulated annealing algorithm proposed
# in the work "On Order Degree Problem for Moore Bound".
#This program is tested and  it correctly implements the proposed algorithm

import numpy as np
import networkx as nx
from itertools import product
from itertools import permutations
import matplotlib.pyplot as plt
import random
import math


#____________INSTRUCTIONS________________________________________________
#1. Run the Algorithm several times to obtain  better results
#2. Change the DEGREE (section 0) according to the degree of the graph 
#3. Parameters in section 8 can be changed and tuned for better results.
#4. Increasing the number of iterations for higher degrees produces better results.

#(0)Initially define the degree of the graph.

DEGREE=5  # degree of the graph (You can change it according to the graph)

n=DEGREE

#___________________________________________________________
#(1) Construct the block D
rows = n + 1
cols = n * (n - 1)
D = np.zeros((rows, cols))

      

for i in range(1, n+1):
    for j in range((i-1)*(n-1), (i-1)*(n-1) + (n-1)):
        D[i,j]=1
D=np.array(D)
#___________________________________________________________


#(2) construct the block L
L= np.zeros((n+1, n+1))

L[0,:]=1
L[:,0]=1
L[0,0]=0

L=np.array(L)
#___________________________________________________________

# (3) Initialize the big matrix P with all permutations are identity
#In this section submatrix P of the adjacency matrix in initialized.

def InitializePermutations(n):
    
    bz = n - 1      #size of a block
    nb = n  # Number of blocks along one dimension
    
    # Initialize an empty block matrix
    Blokmat = []
    for i in range(nb):
        row = []
        for j in range(nb):
            if i == j:
                # Set diagonal blocks to be zero
                row.append(np.zeros((bz, bz)))
            else:
                # Set other blocks to be identity matrix
                row.append(np.eye(bz))
        # Add the blocks
        Blokmat.append(np.hstack(row))
    
    # Concatenate all rows vertically
    final_mat = np.vstack(Blokmat)
    return final_mat

P=InitializePermutations(n) #Initialize P
#___________________________________________________________





# (4) Defining the Energy (objective) Function of the SA Process

def objective(Atemp):
    #This count the number of vertex pairs which break the diameter constraint
    
    G = nx.from_numpy_array(np.array(Atemp))         # Generate the graph G
    
    # All shortest paths
    alp = dict(nx.all_pairs_shortest_path_length(G)) #assign all the shortest path lengths to alp

    n = len(Atemp)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            d = alp.get(i, {}).get(j, float("inf"))
            if d > 2:  
                count += 1
    return count

#___________________________________________________________

#(5) Initialize the adjacency matrix
Adj= np.block([[L, D],          # First row of blocks: L and D
                 [D.T,P] # Second row of blocks: P and transpose of D
                    ])

#___________________________________________________________
# (6) Defining the Swap function (The neibourhood Operator)
def swap(Blokmat, n):
    
    
    Blocksize = n - 1
    Numblox = Blokmat.shape[0] // Blocksize
    
    # Finding all the non diagonal blocks positions
    NonZeroblox = [(i, j) for i in range(Numblox) for j in range(Numblox) if i != j and i!=0 and j!=0]
    
    
    # select one non-zero block randomely
    SelctBlok = random.choice(NonZeroblox)
    Blockrow, Blockol = SelctBlok


    
    
    # Selected block matrix is assigned to a matrix named "SelctBlokMat".
    Rowstart = Blockrow * Blocksize
    Rowend = Rowstart + Blocksize
    Colmnstart = Blockol * Blocksize
    Columnend = Colmnstart + Blocksize
    SelctBlokMat = Blokmat[Rowstart:Rowend, Colmnstart:Columnend]
    
    # Randomly select two rows to swap
    RowIndex = random.sample(range(Blocksize), 2)
    row1, row2 = RowIndex
    
    # Swap the rows in the selected block
    SelctBlokMat[[row1, row2], :] = SelctBlokMat[[row2, row1], :]




    
    # Update the original block matrix with the modified block
    Blokmat[Rowstart:Rowend, Colmnstart:Columnend] = SelctBlokMat
    Blokmat[ Colmnstart:Columnend, Rowstart:Rowend] = SelctBlokMat.T
    
    return Blokmat
#_____________________________________________________________
#_____________________________________________________________
# (7) Simmulated Annealing Algorithm
def simulated_annealing(Adj,iterations, temperature, alpha,P):
    
    #CurrntMatriX = np.array(generate_symmetric_matrix(10, 3))
    CurrntMatriX=Adj
    CurrentFitness = objective(CurrntMatriX)

    Ptest=P
    Pbest=P

    Bestmatrix = CurrntMatriX
    Bestfitness = CurrentFitness

    #temperature = initial_temp
    

    G=nx.Graph(Adj)
    
    
   
   
        
    for iteration in range(iterations):
        # Generate a new candidate solution
        P=swap(Ptest,n)  #Apply the neighbourhood operation
        #P=swap(P,n)
        TestmatriX = np.block([
                                     [L, D],          # First row: A and B
                                     [D.T,P] # Second row: C and transpose of B
                                    ])
       
        

        

        
        candidateFitness = objective(TestmatriX)
       
       
        
        # Calculate acceptance probability
        delta = candidateFitness - CurrentFitness  
        if delta < 0  or random.random() < math.exp(-delta / temperature):
            # Accept the new solution
            CurrntMatriX = TestmatriX
            CurrentFitness = candidateFitness
            Ptest=P

        # Update the best solution found so far
        if CurrentFitness < Bestfitness:
            Bestmatrix = CurrntMatriX
            Bestfitness = CurrentFitness
            Pbest=Ptest
            

       
        
       

        # Cool down the temperature
        temperature *= alpha
        
       # alpha=cooling_rate
        
        #temperature =  temperature / (1 + alpha * np.log(1 + iteration))

    return Bestmatrix, Bestfitness,Pbest
#_________________________________________________________________________________________________

# (8) Define problem parameters [This parameters can be mannually changed]

Iterations = 2500   #Increase  Iterations for better results
Temperature = 2500  #Increasing the temperature , will explore more bad solutions ,
Alpha = 0.99        #Cooling process is controled

#__________________________________________________________________________________________________
print('O N    O R D E R    D E G R E E   P R O B L E M     F O R     M O O R E     B O U N D')
print("_____________________________________________________________________________________")
print("______________________________________________________________")
print("implementation of the first simulated annealing algorithm propesed")
print("_______________________________________________________________")
# (9) Run the simulated Annealing Algorithm

Adj2,Bestfitness,Pbest = simulated_annealing(Adj,Iterations, Temperature, Alpha,P)

G=nx.Graph(Adj2)
diameter=nx.diameter(G) 
print('diameter is ',diameter)
#print (nx.diameter(G))
print('above is the diameter')
Avgdistance = nx.average_shortest_path_length(G)
print(f"Average distance between vertices: {Avgdistance}")

Adjnew=Adj2





#____________________________________________________________________________________________________
# (10) Start the second Run
Iterations = 600
Temperature = 1000
Cooling_rate = 0.99
for i in range (10):
    Adjnew,Bestfitness,Pbest = simulated_annealing(Adjnew,Iterations, Temperature, Cooling_rate,Pbest)
    G=nx.Graph(Adjnew)
    
    print('diameter in',i,'th reheat is',nx.diameter(G))
    #print (nx.diameter(G))
    print('above is the diameter')
    Avgdistance = nx.average_shortest_path_length(G)
    print(f"Average distance between vertices: {Avgdistance}")
    print('Number of excess pairs',Bestfitness)











# (11) print the maximum degree of the graph
print('__________________________________________________________________')
print("Details of the graph are follows")
print('___________________________________________________________________')

M=n**2 * (n**2 + 1) / 2
MDI=Bestfitness/M      #MDI is simplified version for radial Moore graphs
maximum_degree_of_the_graph = max(dict(G.degree()).values())
print('Maximum degree of the graph is,',max(dict(G.degree()).values()))
print('ASPL is',Avgdistance)
print('MDI is',MDI)
print('diameter of the graph is',nx.diameter(G))













