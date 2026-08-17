n = int(input())
cuts = 0
pieces = 1
while pieces < n:
    cuts += 1
    pieces *= 2
print(cuts)
