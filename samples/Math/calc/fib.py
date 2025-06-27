import numpy as np 

def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)
    
print(fib(9)) # 34 = 13 + 21 = 0, 1 -> 1, 2 -> 1, 3 -> 2, 4 -> 3, 5 -> 5, 6 -> 8, 7 -> 13, 8 -> 21, 9 -> 34
    
