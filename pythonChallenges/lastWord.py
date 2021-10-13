def lengthofLastWord(str):
   a = str.strip()
   x = list(a.split(" "))
   return len(x[-1])
   #print("The last word is", x[-1], "with length", len(x[-1]))

s = "Hello World"
print(lengthofLastWord(s))
