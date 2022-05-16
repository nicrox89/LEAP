import zlib
from numpy import linalg as LA
import copy

a = LA.norm([2,8])
c = str(["100","600"]).encode('utf-8')
t = str(["2","88"]).encode('utf-8')

conc = c+t

d = (len(zlib.compress(conc)) - min(len(zlib.compress(c)), len(zlib.compress(t)))) / max(len(zlib.compress(c)), len(zlib.compress(t)))
print(d)