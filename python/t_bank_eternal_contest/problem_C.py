n, t = map(int, input().split())
floors = list(map(int, input().split()))
target_idx = int(input()) - 1
res = 0

time_start_to_target = floors[target_idx] - floors[0]
time_end_to_target = floors[-1] - floors[target_idx]
time_end_to_end = floors[-1] - floors[0]

if  time_start_to_target > t and time_end_to_target > t:
    if time_start_to_target <= time_end_to_target:
        res += time_start_to_target + time_end_to_end
    else:
        res += time_end_to_target + time_end_to_end
else:
    res += time_end_to_end

print(res)