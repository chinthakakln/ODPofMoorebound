
#This provides a python implementation of the latin Square based algorithm proposed
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
#4. Increasing the iterations for higher degrees produces better results.

#First give the degree of the graph


DEGREE=7  #Here the degree of the graph is given. It can be changed and test for different degrees

n=DEGREE-1 
# (1) Construct the Latn Squre.
T=n+1

def check(A,temp,i,j):
  for k in range(i+1):
      for l in range(j+1):
       if k!=0 and l!=0 and k!=i and l!=j and A[i][j]==A[i][k] or A[i][j]==A[k][j]:
        temp=temp+1
        if temp==T:
           temp=1

        else:
         temp=temp%T
        
        A[i][j]=temp
        return A[i][j]
    
A=np.zeros((n+1, n+1))
for j in range (n+1):
    A[0][j]=1
    A[j][0]=1
    A[0,0]=1
for i in range(1,n+1):
 for j in range(i,n+1):
   if j-1==i:
    A[i][j]=A[i-1][j]+(A[1][j]-1)
   else:
    A[i][j]=A[i][j-1]+i

   if A[i][j]==n:
    A[i][j]=n
   else:
    A[i][j]=A[i][j]%n

   A[j][i]=A[i][j]
   A[i][i]=0
   temp=A[i][j]
   for p in range (n+1):
     for k in range(i):
      for l in range(j):
       if (A[i][j]==A[i][l] or A[i][j]==A[k][j]):
        temp=temp+1
        if temp==T:
           temp=1

        else:
         temp=temp%T
        A[i][j]=temp
        A[j][i]=temp
        break
        break
#_____________________________________________________________________________________________________________

n=n+1 # map the input value.
#_____________________________________________________________________________________________________________
# (2) Swaps two rows in a matrix.(This operetion is used later in matrix changing)
def Rowswap(matrix, row1, row2):
    """Swaps two rows in a matrix."""
    matrix[[row1, row2]] = matrix[[row2, row1]]
    return matrix
#_____________________________________________________________________________________________________________
# (3) This generates a permuted identity.
def PermutEYE(n, k):
   
    I = np.eye(n-1)
    
   
    
    I = Rowswap(I, 0, k-1)  # Swap row 1 and row k 
    rows = list(range(n-1))
    rows.remove(0)
    rows.remove(k-1)
    np.random.shuffle(rows)  # Shuffle remaining rows randomly
    
    if (n-1) % 2 == 0:
        for i in range(0, len(rows), 2):
            Rowswap(I, rows[i], rows[i+1])
    else:
        for i in range(0, len(rows)-1, 2):
            Rowswap(I, rows[i], rows[i+1])
        Rowswap(I, rows[-1], np.random.choice([x for x in range(n-1) if x not in [0, k-1, rows[-1]]]))
    
    return I
#___________________________________________________________________________________________________________
# (4) This costruct the initial permutation Block matrix (principle Submatrix based on latin square)

def BlockP(L):
    """Constructs the block matrix P based on matrix L."""
    L = L.astype(int)  # Ensure L contains only integers
    n = L.shape[0]
    Bloksize = n - 1
    P = np.zeros((n * Bloksize, n * Bloksize))

    for i in range(n):
        for j in range(i, n):  # Upper triangular part
            if L[i, j] == 0:
                block = np.zeros((Bloksize, Bloksize))
            elif L[i, j] == 1:
                block = np.eye(Bloksize)
            else:
                block = PermutEYE(n, int(L[i, j]))  # Ensure integer input
            
            P[i*Bloksize:(i+1)*Bloksize, j*Bloksize:(j+1)*Bloksize] = block
            if i != j:
                P[j*Bloksize:(j+1)*Bloksize, i*Bloksize:(i+1)*Bloksize] = block.T

    return P

#____________________________________________________________________________________________________________
# (5) New Swap function (Create random matrix and add a new block)
def swap(P,L,n):
    L = L.astype(int)  # Ensure L contains only integers
    n = L.shape[0]
    Bloksize = n - 1

    i, j = random.sample(range(1, n), 2)  # Generate two distinct random integers

    if L[i, j] == 0:
        block = np.zeros((Bloksize, Bloksize))
    elif L[i, j] == 1:
        block = np.eye(Bloksize)
    else:
        block = PermutEYE(n, int(L[i, j]))  # Ensure integer input
    
    P[i*Bloksize:(i+1)*Bloksize, j*Bloksize:(j+1)*Bloksize] = block
    if i != j:
        P[j*Bloksize:(j+1)*Bloksize, i*Bloksize:(i+1)*Bloksize] = block.T

    return P

#________________________________________________________________________________________________________________
# (6) Initialization of the blocks and the adjacency matrix
# Convert A to integers before passing it to the function
A = A.astype(int)
P = BlockP(A)




rows = n + 1
columns = n * (n - 1)
D = np.zeros((rows, columns))

for i in range(1, n+1):
    for j in range((i-1)*(n-1), (i-1)*(n-1) + (n-1)):
        D[i,j]=1
D=np.array(D)
print(D)
        
      
L= np.zeros((n+1, n+1))

L[0,:]=1
L[:,0]=1
L[0,0]=0

L=np.array(L)
Adj= np.block([[L, D],          # First row: A and B
                 [D.T,P] # Second row: C and transpose of B
                    ])
#________________________________________________________________________________________________________________
# (6) Defining the Energy Function of the SA Process

def objective(A):
    # Create an undirected graph from the adjacency matrix
    G = nx.from_numpy_array(A)

    count = 0
    nodes = list(G.nodes())

    # Compute all-pairs shortest path lengths
    lengths = dict(nx.all_pairs_shortest_path_length(G))

    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):  # count each pair once
            d = lengths[nodes[i]].get(nodes[j], float('inf'))  # distance or ∞ if disconnected
            if d > 2:
                count += 1

    return count




#______________________________________________________________________________________________________________________
# (7) Simulated Annealing Algorithm

def simulated_annealing(Adj,Iterations, Temperature0, ALPHA,P,A,objectives,iterations,itt,bests):
    
    #objectives=[]
    #iterations=[]
    #itt=0
    #CurntMatrX = np.array(generate_symmetric_matrix(10, 3))
    CurntMatrX=Adj
    Currentfitness = objective(CurntMatrX)

    PCurrent=P
    Pbest=P

    BestMatrX = CurntMatrX
    Bestfitness = Currentfitness

    temperature = Temperature0
    

    G=nx.Graph(Adj)
    
    
   
   
        
    for iteration in range(Iterations):
        # Generate a new candidate solution
        P=swap(PCurrent,A,n)
        #P=swap(P,n)
        TestMatriX = np.block([
                                     [L, D],          # First row: A and B
                                     [D.T,P] # Second row: C and transpose of B
                                    ])
       
        

        

        
        Testfitness = objective(TestMatriX)
        objectives.append(Currentfitness)
        iterations.append(itt)
        bests.append(Bestfitness)
        itt=itt+1
        
        
        # Calculate acceptance probability
        delta = Testfitness - Currentfitness  
        if delta < 0  or random.random() < math.exp(-delta / temperature):
            # Accept the new solution
            CurntMatrX = TestMatriX
            Currentfitness = Testfitness
            PCurrent=P
            #objectives.append(Currentfitness)
            #iterations.append(itt)
            #bests.append(Bestfitness)
            #itt=itt+1
            

        # Update the best solution found so far
        if Currentfitness < Bestfitness:
            BestMatrX = CurntMatrX
            Bestfitness = Currentfitness
            Pbest=PCurrent
            
       
        
       

        # Cool down the temperature
        temperature *= ALPHA
        
        alpha=ALPHA
        
        #temperature =  temperature / (1 + alpha * np.log(1 + iteration))

    return BestMatrX, Bestfitness,Pbest,objectives,iterations,itt,bests  
 #________________________________________________________________________________________________--

# Construct the intial Graps

G2=nx.Graph(Adj)
#nx.draw(G)
#plt.show()
print('diameter is follows')
print (nx.diameter(G2))
print('above is the diameter')
AvgDist = nx.average_shortest_path_length(G2)
print(f"Average distance between vertices: {AvgDist}")

#_____________Check the latin squre for degree 11 __________________

#Following latin square may manually added if degree 11 (This latin squre is not generated from the methods we used)
#The program will give good results even with the usual method for degree 11.
"""A = [
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 2],
    [1, 3, 0, 2, 6, 7, 5, 9, 10, 8, 4],
    [1, 4, 2, 0, 7, 5, 6, 10, 8, 9, 3],
    [1, 5, 6, 7, 0, 9, 10, 2, 3, 4, 8],
    [1, 6, 7, 5, 9, 0, 8, 3, 4, 2, 10],
    [1, 7, 5, 6, 10, 8, 0, 4, 2, 3, 9],
    [1, 8, 9, 10, 2, 3, 4, 0, 6, 7, 5],
    [1, 9, 10, 8, 3, 4, 2, 6, 0, 5, 7],
    [1, 10, 8, 9, 4, 2, 3, 7, 5, 0, 6],
    [1, 2, 4, 3, 8, 10, 9, 5, 7, 6, 0]
]
A=np.array(A)"""

print('O N    O R D E R    D E G R E E   P R O B L E M     F O R     M O O R E     B O U N D')
print("_____________________________________________________________________________________")
print("______________________________________________________________")
print("implementation of the Latin square algorithm propesed")
print("_______________________________________________________________")
#______________________________________________________________
Iterations=500      #These parameters can be changed and tuned 
Temperature0=700
ALPHA=0.98



objectives=[]
iterations=[]
bests=[]
itt=0
Adj2,count,P_new,objectives,iterations,itt,bests=simulated_annealing(Adj,Iterations, Temperature0, ALPHA,P,A,objectives,iterations,itt,bests)

plt.plot(iterations, objectives, linestyle='-', label='Current Energy')
plt.plot(iterations, bests, color='red', linestyle='--', label='Best Energy')

plt.xlabel('Iterations')
plt.ylabel('Energy')
plt.title('Energy vs Iterations – Simulated Annealing on Latin Square')
plt.grid(True)
plt.legend()
plt.show() 
Iterations=500
Temperature0=700
ALPHA=0.99
for i in range (10):
    Adj2,count,P_new,objectives,iterations,itt,bests=simulated_annealing(Adj2,Iterations, Temperature0, ALPHA,P_new,A,objectives,iterations,itt,bests)
    
      
      
    print("following result is after runnun SA Algorithm")
    G2=nx.Graph(Adj2)
    G2.remove_edges_from(nx.selfloop_edges(G2)) # remove if there are self loops
    #nx.draw(G)
    #plt.show()
    print('diameter is follows')
    print (nx.diameter(G2))
    print('above is the diameter')
    AvgDist = nx.average_shortest_path_length(G2)
    print(f"Average distance between vertices: {AvgDist}")
    Maxdeg = max(dict(G2.degree()).values())
    print("Maximum degree of the graph:", Maxdeg)
    print("Self-loops:", list(nx.selfloop_edges(G2)))


plt.plot(iterations, objectives, linestyle='-', label='Current Energy')
plt.plot(iterations, bests, color='red', linestyle='--', label='Best Energy')

plt.xlabel('Iterations')
plt.ylabel('Energy')
plt.title('Energy vs Iterations – Simulated Annealing on Latin Square')
plt.grid(True)
plt.legend()
plt.show()   

#___-Simmulated Annealing with row swap___

print("simulated Annealing with row swap")
print("________________________________")
print("_______________________________")

def rowswap(BlokmatriX, n):
    
    Bloksize = n - 1
    num_blocks = BlokmatriX.shape[0] // Bloksize
    
    # Find all off-diagonal block indices
    NZBlocks = [(i, j) for i in range(num_blocks) for j in range(num_blocks) if i != j and i!=0 and j!=0]
    
    # Randomly select one non-zero block
    Blokselect = random.choice(NZBlocks)
    Brow, Bcol = Blokselect
    
    # Extract the selected block
    Rowstart = Brow * Bloksize
    Rowend = Rowstart + Bloksize
    ColumnStart = Bcol * Bloksize
    ColumnEnd = ColumnStart + Bloksize
    SelectedBlokMatriX = BlokmatriX[Rowstart:Rowend, ColumnStart:ColumnEnd]
    
    # Randomly select two rows to swap
    row_indices = random.sample(range(Bloksize), 2)
    row1, row2 = row_indices
    
    # Swap the rows in the selected block
    SelectedBlokMatriX[[row1, row2], :] = SelectedBlokMatriX[[row2, row1], :]
    
    # Update the original block matrix with the modified block
    BlokmatriX[Rowstart:Rowend, ColumnStart:ColumnEnd] = SelectedBlokMatriX
    BlokmatriX[ ColumnStart:ColumnEnd, Rowstart:Rowend] = SelectedBlokMatriX.T
    
    return BlokmatriX
      


def simulated_annealing2(Adj,Iterations, Temperature0, ALPHA,P):
    
    #CurntMatrX = np.array(generate_symmetric_matrix(10, 3))
    CurntMatrX=Adj
    Currentfitness = objective(CurntMatrX)

    BestMatrX = CurntMatrX
    Bestfitness = Currentfitness

    temperature = Temperature0
    
    
    
   
    
    #testing Code
    PCurrent=P
    Pbest=P
    for iteration in range(Iterations):
        # Generate a new candidate solution
        P=rowswap(PCurrent,n)
        TestMatriX = np.block([
                                     [L, D],          # First row: A and B
                                     [D.T,P] # Second row: C and transpose of B
                                    ])
       
        

        

        #Testfitness = objective2(TestMatriX,D,L)
        Testfitness = objective(TestMatriX)
        #Testfitness=ObjectiveD(7,P)

        # Calculate acceptance probability
        delta = Testfitness - Currentfitness  
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            # Accept the new solution
            CurntMatrX = TestMatriX
            Currentfitness = Testfitness
            PCurrent=P

        # Update the best solution found so far
        if Currentfitness < Bestfitness:
            BestMatrX = CurntMatrX
            Bestfitness = Currentfitness
            Pbest=PCurrent
        

        # Cool down the temperature
        temperature *= ALPHA
        
        alpha=ALPHA
        
        #temperature =  temperature / (1 + alpha * np.log(1 + iteration))

    return BestMatrX, Bestfitness, Pbest

Adjn=Adj2

Iterations = 1000
Temperature0 = 2000
ALPHA = 0.99
#print(generate_symmetric_matrix(10, 3))
# Run simulated annealing
for i in range(2):
  Adjn, count,P_new = simulated_annealing2(Adjn,Iterations, Temperature0, ALPHA,P_new)
  G3=nx.Graph(Adjn)
  G3.remove_edges_from(nx.selfloop_edges(G3)) # remove if there are self loops
  print ("The diameter is",nx.diameter(G3))
  AvgDist = nx.average_shortest_path_length(G3)
  print("average shortest path length of",i,"th output is --->",AvgDist)


G3=nx.Graph(Adjn)
G3.remove_edges_from(nx.selfloop_edges(G3)) # remove if there are self loops

print('diameter is follows')
print (nx.diameter(G3))
print('above is the diameter')
AvgDist = nx.average_shortest_path_length(G3)
print(f"Average distance between vertices: {AvgDist}")
Maxdeg = max(dict(G3.degree()).values())
print("Maximum degree of the graph:", Maxdeg)
print("Self-loops:", list(nx.selfloop_edges(G3)))

#_____________Details if the Final Graph

shortest_paths = dict(nx.all_pairs_shortest_path_length(G3))

# Count pairs based on the shortest distance
exceeds_2 = 0
does_not_exceed_2 = 0

# Iterate through all vertex pairs
for u in shortest_paths:
    for v, dist in shortest_paths[u].items():
        if u < v:  # Consider each pair once
            if dist > 2:
                exceeds_2 += 1
            else:
                does_not_exceed_2 += 1

# Print the results
print(f"Number of vertex pairs with distance > 2: {exceeds_2}")
print(f"Number of vertex pairs with distance <= 2: {does_not_exceed_2}")

Bestfitness=exceeds_2
#_______________Print the details of the graph____________________
print('__________________________________________________________________')
print('__________________________________________________________________')
print('See the summery of the results below')
print('_________________________________________')
m=DEGREE
M=m**2 * (m**2 + 1) / 2
MDI=Bestfitness/M
maximum_degree_of_the_graph = max(dict(G3.degree()).values())
print('Maximum degree of the graph is,',max(dict(G3.degree()).values()))
print('ASPL is',AvgDist)
print('MDI is',MDI)
print('diameter of the graph is',nx.diameter(G3))

print('_________________________________________')

np.set_printoptions(threshold=np.inf)  # Print the full matrix
print(Adjn)


