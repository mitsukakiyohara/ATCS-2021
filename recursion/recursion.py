#Recursion Exercise 
#1: Factorial
def factorial(n):
    if n == 0: 
        return 1
    return n * factorial(n - 1)

#2: Countdown 
def countdown(a): 
    if a == 0: 
        return "Blastoff!"
    return str(a) + " " + countdown(a - 1)

#3: Reverse a String
def reverse(s):
    if len(s) == 1: 
        return s
    return s[-1] + reverse(s[:-1])

#4: Bacteria 
def bacteria(n): 
    if n == 0: 
        return 10
    return 2 * bacteria(n - 1) + bacteria(n - 1)

