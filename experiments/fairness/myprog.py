#from TestSuite_Gen import TestSuite_Zero
import pickle
import sys


# def my_partial_fn(x):
#     if x>10:
#         y = 10
#     else:
#         y = 0
#     return y

# my_partial_fn(1)
# my_partial_fn(11)

#female=1 male=0
def decide(applicant):
    gender = 1
    if applicant[gender] == 1:
        y = 0
    else:
        y = 1
    print(y)


#Load the variable
TestSuite_Zero = pickle.load(open("variableStoringFile.dat", "rb"))
#TestSuite_Zero = sys.argv
print(TestSuite_Zero)

for i in range(len(TestSuite_Zero)-1):
    decide(TestSuite_Zero[i])

# coverage run --branch myprog.py
# coverage report -m
# coverage report -m myprog.py
# coverage html
# open htmlcov/myprog_py.html
# x=$(python TestSuite_Gen.py)