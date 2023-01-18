from src.intro import main

# main()
import numpy as np

def my_func(a):
    """Average first and last element of a 1-D array"""
    print(a)
    print(f"a[0]:{a[0]}")
    print(f"a[-1]:{a[-1]}")
    print('')
    return (a[0] + a[-1]) * 0.5

b = np.array([[[1,2,3], [4,5,6], [7,8,9]],
             [[1,2,3], [4,5,6], [7,8,9]]])
# print(b.shape)

# If Axis = 0, this will operate and slice the array along the row
ROW = 0
# Else this will operate and slice the array along the column
COL = 1
print(np.apply_along_axis(my_func, axis=2, arr=b))