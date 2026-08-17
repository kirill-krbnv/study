n, k = map(int, input().split())
nums = input().split()

def my_sort(nums):
    nums = sorted(nums, key=lambda x: x, reverse=False)
    nums = sorted(nums, key=len, reverse=True)
    return nums

nums = my_sort(nums)

n_oper = 0
total = 0
current_len = len(nums[0])

while n_oper < k and current_len >= 1:
    i = 0
    while i < len(nums) and len(nums[i]) == current_len and n_oper < k:
        n = nums[i][0]
        if n != '9':
            total += (9 - int(n)) * int('1' + (len(nums[i])-1) * '0')
            nums[i] = nums[i][1:]
            i += 1
            n_oper += 1
        else:
            nums[i] = nums[i][1:]
            i += 1

    nums = my_sort(nums)
    current_len -= 1

print(total)