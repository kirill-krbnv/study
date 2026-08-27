from collections import Counter

n, m = map(int, input().split())
matrix = []
for _ in range(n):
    matrix.append(list(map(int, input().split())))

total = 0
for i in range((n + 1) // 2):
    for j in range((m + 1) // 2):
        positions = {(i, j), (i, m - 1 - j), (n - 1 - i, j), (n - 1 - i, m - 1 - j)}
        values = [matrix[x][y] for x, y in positions]
        counts = Counter(values)
        total += len(values) - max(counts.values())

print(total)