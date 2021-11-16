import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the data
df = pd.read_csv("../data/oldFaithful.csv")

# Set values for x and y
x = df["wait_time"]
y = df["eruption_time"]

# Determine correlation
correlation = y.corr(x)
print("Correlation: " + str(correlation))

# Add labels
plt.title("Old Faithful Eruptions")
plt.xlabel("Wait Time")
plt.ylabel("Eruption Time")

# Plot the scatterplot
plt.scatter(x, y)

# Create the model
model = np.polyfit(x, y, 1)
print(model)

# Print the line of best fit
m = str(round(model[0], 2))
b = str(round(model[1], 2))

print("Model: y = " + m +"x + " + b)

x_lin_reg = range(df.wait_time.min(), df.wait_time.max())

# Predict using the model
predict = np.poly1d(model)

# Set the y-values of the line of best fit
y_lin_reg = predict(x_lin_reg)

# Plot the line of best fit
plt.plot(x_lin_reg, y_lin_reg, color = "red")

plt.show()

'''-----------Seaborn Scatter Plot-----------'''

ax = sns.regplot(data=df, x="wait_time", y="eruption_time")
ax.set(xlabel='Wait Times', ylabel='Eruption Times', title='Old Faithful Eruptions')

plt.show()