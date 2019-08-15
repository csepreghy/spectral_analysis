import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

with open('../data/FinalTable_10-15-Zoe.pkl', 'rb') as f:

    x1 = pickle.load(f)

df=pd.DataFrame(x1)

# plt1.figure(figsize=(16, 10))

plt.style.use('dark_background')

plt.figure(figsize=(16, 10))

plt2 = sns.boxplot(x="class", y="z", data=df)
plt2.figure.savefig("../plots/boxplot.png")

plt3 = df['class'].value_counts().plot(kind='bar')
plt3.figure.savefig("../plots/histogram.png")

# plt4=sns.swarmplot(y=df['z'], x=df['class'])
# plt4.figure.savefig("../plots/swarmplot.png")
# plt.show()

# plt5=sns.jointplot(x='zErr', y='z', data=df, kind='kde')
# plt5.savefig("../plots/joinplot.png")

plt1 = sns.catplot(x="class", y="z", data=df)
plt1.savefig("../plots/catplot.png")

# Plot miles per gallon against horsepower with other semantics
plt6 = sns.relplot(x="ra", y="dec", hue="class", size="z",sizes=(40, 400)
                 , alpha=.5, palette="muted",
            height=6, data=df)
# plt6 =sns.scatterplot(x='ra', y='dec', hue='class',size='z', data=df ) #,style='class'
# plt6.grid(True)
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt6.savefig("../plots/scatterplot.png")


# plt.show()