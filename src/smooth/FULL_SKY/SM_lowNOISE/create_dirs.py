import os

# for i in range(100):
#     os.makedirs(str(i))

for i in range(100):
    for folder in ['T', 'E', 'B']:
        os.makedirs(f"{i}/{folder}")


