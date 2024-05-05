import os
tt=0
while True:
    os.system('python gen.py > A.in')
    os.system('./a < A.in > a.out')
    os.system('./b < A.in > b.out')
    if os.system('diff a.out b.out'):
        print("WA")
        exit(0)
    else:
        tt+=1
        print("AC:",tt)