from random import *
n = 10000
s = 'qwertyuiopasdfghjklzxcvbnm'
for i in range(n):
    print(choice(s), end = '')
print()
print(randint(0, 1), randint(1, n))