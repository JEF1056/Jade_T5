import numpy as np

lst=['pooh', 'rabbit', 'piglet', 'Christopher', "james", "robert"]

e=np.sort(np.random.choice(len(lst)-1,2, replace=False))
print(e)
print('\t'.join(lst[e[0]:e[1]]))
print(lst[e[1]])