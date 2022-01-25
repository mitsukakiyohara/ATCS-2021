import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("swimTimes.csv")

x = df["year"]
y = df["minutes"]

correlation = y.corr(x)
print("Correlation: " + str(correlation))

plt.title("Swim Times")
plt.xlabel("Year")
plt.ylabel("Minutes")

plt.scatter(x, y)

model = np.polyfit(x, y, 1) #deg = 1
print(model)

m = str(round(model[0], 3))
b = str(round(model[1], 3))

print("Model: y = " + m +"x + " + b)

x_lin_reg = range(df.year.min(), df.year.max())

# Predict using the model
predict = np.poly1d(model)

# Set the y-values of the line of best fit
y_lin_reg = predict(x_lin_reg)

# Plot the line of best fit
plt.plot(x_lin_reg, y_lin_reg, color = "red")

plt.show()

'''-----------Seaborn Scatter Plot-----------

ax = sns.regplot(data=df, x="year", y="minutes")
ax.set(xlabel='Year', ylabel='Minutes', title='Swim Times')

plt.show()
'''
