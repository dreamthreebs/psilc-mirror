import pickle

with open('./ps_list.pkl', 'rb') as f:
    loaded_list = pickle.load(f)
print(loaded_list)  # Output: [14, 6, 8]

