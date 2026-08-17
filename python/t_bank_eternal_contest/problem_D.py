# получение данных
n, k = map(int, input().split())
nums = list(map(int, input().split()))
# подготовка данных
before = sum(nums)
nums = sorted(nums, reverse=True)
nums = list(map(str, nums))

# замена всех цифр на 9, начиная с наибольшего числа, с наивысших разрядов
n_oper = 0
order = -len(nums[0])
while n_oper <= k and order <= -1:
    i = 0
    while len(nums[i]) >= -order and n_oper < k and i <= len(nums):
        if nums[i][order] != '9':
            if len(nums[i]) == 1:
                nums[i] = '9'
            else:
                nums[i] = nums[i][:order] + "9" + nums[i][order + 1 :]
            n_oper += 1
        i += 1
    order += 1
# проверка полученного списка
# переход от строк к числам
nums = map(int, nums)
# нахождение суммы элементов итогового массива
after = sum(nums)
# нахождение результата
res = after - before

print(res)