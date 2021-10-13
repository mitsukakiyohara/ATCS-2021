def validPalindrome(s):
    l = []
    x = s.split(" ")
    
    for i in range(len(s)):
        if s[i].isalnum():
            l.append(s[i].lower())
    
    return l == l[::-1]

s = "A man, a plan, a canal: Panama"
print(validPalindrome(s))


