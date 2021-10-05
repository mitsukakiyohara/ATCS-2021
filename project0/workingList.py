careers = ['programmer', 'scientist', 'journalist', 'engineer']
print(careers.index('programmer'))
print('programmer' in careers)
careers.append('doctor')
careers.insert(0, 'lawyer')

for c in careers:
    print(c)
