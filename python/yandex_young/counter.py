m = int(input())
a = int(input())
b = int(input())
res = 0

if a < b:
    res = b - a

elif a == b:
    res = 0

else:
    res = m - a + b

print(res)
