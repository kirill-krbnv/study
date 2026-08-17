a, b, c, d = map(int, input().split())

if d >= b:
    cost = a + (d - b) * c

else:
    cost = a
print(cost)