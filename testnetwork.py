import sys
print(sys.path)
import os
print(os.getcwd())
try:
    import pynet
    print("Import successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    # Let's see what files are in the current directory
    print("Files in current directory:")
    print(os.listdir('.'))
mat = pynet.Matrix(10, 10)
print(mat.getRow())
print(mat.getCol())