import lmdb
import sys

env = lmdb.open(sys.argv[1], readonly=True)

print(env.stat())
