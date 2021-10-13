#input 
nums = [3, 1, 2, 10, 1]

def runningSum(nums):
    runningSum = [None] * len(nums)
    sum = 0
    for i in range(0, len(nums)): 
        sum = nums[i] + sum
        runningSum[i] = sum
    return runningSum
print("input", nums)
print("output", runningSum(nums))
