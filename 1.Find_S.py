#Implement and demonstrate the Find-S algorithm for finding the most specific hypothesis

import csv

num_attributes = 6
a = []
print("\n The Given Training Data Set \n")

with open(r'D:/My Files/Engineering/6th Sem/Machine Learning Lab/Dataset/Data.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    for row in reader:
        a.append(row)
        print(row)

print("\n The initial value of hypothesis: ")

hypothesis = ['0'] * num_attributes
print(hypothesis)
for j in range(0, num_attributes):
    hypothesis[j] = a[0][j]

print("\n Find S: Finding a Maximally Specific Hypothesis\n")

for i in range(0, len(a)):
    if a[i][num_attributes] == 'Yes':
        for j in range(0, num_attributes):
            if a[i][j] != hypothesis[j]:
                hypothesis[j] = '?'
            else:
                hypothesis[j] = a[i][j]
    print(" For Training instance No:{0} the hypothesis is".format(i), hypothesis)

print("\n The Maximally Specific Hypothesis for a given TrainingExamples :\n")
print(hypothesis)


# Output:

# The Given Training Data Set 

# ['Sunny ', 'Warm ', 'Normal', 'Strong', 'Warm', 'same', 'yes']
# ['Sunny ', 'Warm ', 'High', 'Strong', 'Warm', 'same', 'yes']
# ['Rainy ', 'Cold', 'High', 'Strong', 'Warm', 'change', 'no ']
# ['Sunny ', 'Warm ', 'High', 'Strong', 'Cool', 'change', 'yes']

#  The initial value of hypothesis: 
# ['0', '0', '0', '0', '0', '0']

#  Find S: Finding a Maximally Specific Hypothesis

#  For Training instance No:0 the hypothesis is ['Sunny ', 'Warm ', 'Normal', 'Strong', 'Warm', 'same']
#  For Training instance No:1 the hypothesis is ['Sunny ', 'Warm ', '?', 'Strong', 'Warm', 'same']
#  For Training instance No:2 the hypothesis is ['Sunny ', 'Warm ', '?', 'Strong', 'Warm', 'same']
#  For Training instance No:3 the hypothesis is ['Sunny ', 'Warm ', '?', 'Strong', '?', '?']

#  The Maximally Specific Hypothesis for a given TrainingExamples :

# ['Sunny ', 'Warm ', '?', 'Strong', '?', '?']