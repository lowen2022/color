import colour
import numpy as np
a=[1,2,3]
b=[4,5,6,7,8,9]
print(b[2:5])
c=[x*y for x,y in zip(a,b)]
print(c)
print(colour.__version__)
