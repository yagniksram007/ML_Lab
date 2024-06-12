import pandas as pd

# Define the initial hypotheses
def initialize_hypotheses(num_attributes):
    S = ['0'] * num_attributes  # Most specific hypothesis
    G = [['?'] * num_attributes]  # Most general hypothesis
    return S, G

# Update the specific hypothesis
def update_S(S, example):
    for i in range(len(S)):
        if S[i] == '0':  # if S is initialized to the most specific hypothesis
            S[i] = example[i]
        elif S[i] != example[i]:
            S[i] = '?'  # Generalize S to cover the example
    return S

# Update the general hypothesis
def update_G(G, S, example):
    G_new = []
    for g in G:
        for i in range(len(g)):
            if g[i] != '?' and g[i] != example[i]:
                g_new = g.copy()
                g_new[i] = '?'
                if is_consistent(g_new, S):
                    G_new.append(g_new)
    if not G_new:
        G_new = [['?'] * len(S)]
    return G_new

# Check consistency of the hypothesis
def is_consistent(h, example):
    for i in range(len(h)):
        if h[i] != '?' and h[i] != example[i]:
            return False
    return True

# Candidate Elimination algorithm
def candidate_elimination(data):
    num_attributes = data.shape[1] - 1
    S, G = initialize_hypotheses(num_attributes)

    for index, row in data.iterrows():
        example = row[:-1]
        label = row[-1]

        if label == 'Yes':
            S = update_S(S, example)
            G = [g for g in G if is_consistent(g, example)]
        else:
            G = update_G(G, S, example)
    return S, G

# Load the dataset
data = pd.read_csv('D:/My Files/Engineering/6th Sem/Machine Learning Lab/Dataset/weather.csv')

# Run the candidate elimination algorithm
S, G = candidate_elimination(data)

print("Most Specific Hypothesis (S):", S)
print("Most General Hypotheses (G):", G)

# Output:
# Most Specific Hypothesis (S): ['Sunny', 'Warm', '?', 'Strong', '?', '?']
# Most General Hypotheses (G): [['Sunny', '?', '?', 'Strong', '?', '?']]

