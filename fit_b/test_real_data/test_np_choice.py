import numpy as np

array = [1, 2, 3, 4, 5]
choice = np.random.choice(array)
print(choice)

array = [1, 2, 3, 4, 5]
choices = np.random.choice(array, size=3)
print(choices)


array = [1, 2, 3, 4, 5]
choices = np.random.choice(array, size=3, replace=False)
print(choices)

array = [1, 2, 3, 4, 5]
probabilities = [0.1, 0.2, 0.3, 0.2, 0.2]
choices = np.random.choice(array, size=3, p=probabilities)
print(choices)
