import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/')

from google.colab import drive


import csv
def get_data(split):
    data_path =root +"QM9_Smiles_sorted.csv"
    with open(data_path) as f:
        data = csv.reader(f)
    
        # Skip header
        next(data)
        
        # Get smiles and targets
        smiles, Y = [], []
        count=0
        for row in data:
            if (count<60000):
              smiles.append(row[12])
              Y.append(row)
              count=count+1
    
    return smiles, Y

AllSmiles, All = get_data('train')

"""## Data Processing"""

# Define vocab
vocab = {char for smiles in AllSmiles for char in smiles}

print(f'Vocab = {vocab}')

# Create word to index mapping
padding_idx = 0
char_to_index = {char: index + 1 for index, char in enumerate(vocab)}
vocab_size = len(char_to_index) + 1

print(f'Vocab size = {vocab_size:,}')

char_to_index

"""## Map Characters to Indices"""

biggest_mol_size = max([len(smiles) for smiles in AllSmiles])
print("Total number of smiles= ", len(AllSmiles))
print("size of largest molecule = ", biggest_mol_size)

X = [[char_to_index[char] for char in smiles] for smiles in AllSmiles]

"""## Add Padding"""

#add all elemnets the pad with os at the end so that they have the same length
max_len = 25
Smiles = [seq[:max_len] + [padding_idx] * (max_len - len(seq)) for seq in X]

print(f'Smiles string = {AllSmiles[0]}')
print(f'Indices of first train SMILES = {Smiles[0]}')
print(f'Last five indices = {Smiles[0][-5:]}')