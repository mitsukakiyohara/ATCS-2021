mountains = {'Mount Everest': 8848, 'K2': 8611, 'Kangchenjunga': 8586, 'Lhoste': 8516, 'Makulu': 8485}

#for key_name in mountains.keys(): 
#   print(key_name)

#for value_name in mountains.values(): 
#    print(value_name)

for key_name, value_name in mountains.items():
    print(key_name, "is", value_name, "meters tall.")