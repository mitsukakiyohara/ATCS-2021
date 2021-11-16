import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

df = pd.read_csv('marvel.csv')
#print(df.info())

df["Total"] = df["Intelligence"] + df["Speed"] + df["Durability"] + df["Power"] + df["Combat"]
#print(df["Total"])

#Pie Chart
df1 = df.groupby("Stamina").sum()
df2 = df.groupby("Intelligence").sum()
#print(df1)

#df1.plot.pie(y="Combat", labels=df1.index, title="Combat by Level of Stamina")
df1.plot.pie(y="Combat", labels=df1.index, title="Combat by Level of Stamina",figsize=(5, 5),colors = ['red', 'pink'])
plt.show()
plot = df2.plot.pie(subplots=True, figsize=(40, 20))

#Histogram
#df["Intelligence"].plot(kind="hist", title="Intelligence Histogram")
df["Intelligence"].plot(kind="hist", title="Intelligence", bins=6,color="green")
df["Intelligence"].plot.hist(orientation="horizontal", cumulative=True,grid=True);
plt.show()

#Scatterplot
#df.plot.scatter(x="Speed", y="Durability", title="Durability vs. Speed")
df.plot.scatter(x="Speed", y="Durability", title="Durability vs. Speed", 
                c="Red", s=30, marker="v", edgecolors="black", figsize=(10,5))

#obtain m (slope) and b(intercept) of linear regression line
#m, b = np.polyfit(x=df["Speed"], y=df["Durability"], deg=1)
#add linear regression line to scatterplot 
#plt.plot(x, m*x+b)
#plt.show()
