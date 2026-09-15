from sys import stdin

n, m = list(map(int, input().split()))
row_res = []
col_res = [[] for _ in range(m)]
prev_state = [False for _ in range(m)]
col_lens = [0 for _ in range(m)]
for line in stdin:
    row = list(line)
    # первым делом обработаем строку по горизонтали
    cur_row_res = [] # -> получаем список длин отрезков в строке
    prev_black = False
    segment_len = 0
    # обработка столбцов
    cur_state = [False for _ in range(m)]
    for i in range(m):
        if row[i] == '#':
            segment_len += 1
            prev_black = True
            # обработка столбцов
            cur_state[i] = True
            col_lens[i] += 1
        else:
            if prev_black:
                cur_row_res.append(segment_len)
                segment_len = 0
                prev_black = False
            # обработка столбцов
            if prev_state[i]:
                col_res[i].append(col_lens[i])
                col_lens[i] = 0
    if prev_black:
        cur_row_res.append(segment_len)
    prev_state = cur_state
    # добавяем в строковый результат список длин черных отрезков
    row_res.append(cur_row_res)
    # теперь обработаем каждый столбец
# в конце выполнения внешнего цикла нужно добавить все значения длин отрезков по вертикали,
# которые дошли до конца и не были добавлены в результат
for i in range(m):
    if col_lens[i] != 0:
        col_res[i].append(col_lens[i])

for r in row_res:
    if not r:
        print(0)
    else:
        print(len(r), *r, sep=' ')
print()
for c in col_res:
    if not c:
        print(0)
    else:
        print(len(c), *c, sep=' ')