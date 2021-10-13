def twoSum(nums, target):
    n = len(nums)
    d = {}
    for i in range(n):
        rem = target - nums[i]  # remaining value
        if rem in d:            # remaining value found
            return [d[rem],i]   # returning the value present at key with the current index
        else:
            d[nums[i]] = i    # updating the dict. with key as number as value as index

l = [2, 7, 11, 15]
print(twoSum(l, 9))

