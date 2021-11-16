import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('marvel.csv')
print(df)

combat = pd.Series(df['Combat'])
plt.boxplot(combat)
plt.title("Distribution of Combat")
plt.show()








