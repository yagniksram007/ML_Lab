#Implement and demonstrate the candidate elimination algorithm using dataset stored as a CSV.file

import numpy as np 
import pandas as pd

data = pd.read_csv('D:/My Files/Engineering/6th Sem/Machine Learning Lab/Dataset/Data.csv')
concepts = np.array(data.iloc[:,0:-1])
target = np.array(data.iloc[:,-1])  
def learn(concepts, target): 
    specific_h = concepts[0].copy()  
    print("\nInitialization of specific_h \n",specific_h)  
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]     
    print("\nInitialization of general_h \n", general_h)  

    for i, h in enumerate(concepts):
        if target[i] == "yes":
            print("If instance is Positive ")
            for x in range(len(specific_h)): 
                if h[x]!= specific_h[x]:                    
                    specific_h[x] ='?'                     
                    general_h[x][x] ='?'
                   
        if target[i] == "no":            
            print("If instance is Negative ")
            for x in range(len(specific_h)): 
                if h[x]!= specific_h[x]:                    
                    general_h[x][x] = specific_h[x]                
                else:                    
                    general_h[x][x] = '?'        

        print(" step {}".format(i+1))
        print(specific_h)         
        print(general_h)
        print("")

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]    
    for i in indices:   
        general_h.remove(['?', '?', '?', '?', '?', '?']) 
    return specific_h, general_h 

s_final, g_final = learn(concepts, target)

print("Final Specific_h:", s_final, sep="\n")
print("Final General_h:", g_final, sep="\n")


# Output: 

# Initialization of specific_h 
#  ['Sunny' 'Warm' 'Normal' 'Strong' 'Warm' 'Same']

# Initialization of general_h 
#  [['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]
#  step 1
# ['Sunny' 'Warm' 'Normal' 'Strong' 'Warm' 'Same']
# [['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]

#  step 2
# ['Sunny' 'Warm' 'Normal' 'Strong' 'Warm' 'Same']
# [['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]

#  step 3
# ['Sunny' 'Warm' 'Normal' 'Strong' 'Warm' 'Same']
# [['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]

#  step 4
# ['Sunny' 'Warm' 'Normal' 'Strong' 'Warm' 'Same']
# [['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]

# Final Specific_h:
# ['Sunny' 'Warm' 'Normal' 'Strong' 'Warm' 'Same']
# Final General_h:
# []