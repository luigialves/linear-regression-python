import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

dataframe = pandas.read_csv('sample.csv')

print(dataframe)

pageviews = dataframe.pageviews[:, np.newaxis]

pageviews_train = pageviews[:-20]

pageviews_test = pageviews[-20:]

materias_train = dataframe.materias[:-20]

materias_test = dataframe.materias[-20:]

regr = linear_model.LinearRegression()

regr.fit(pageviews_train, materias_train)

print('Coefficients: \n', regr.coef_)

print("Mean squared error: %.2f"
      % np.mean((regr.predict(pageviews_test) - materias_test) ** 2))

print('Variance score: %.2f' % regr.score(pageviews_test, materias_test))

plt.scatter(pageviews_test, materias_test, color='black')

plt.plot(pageviews_test, regr.predict(pageviews_test), color='blue', linewidth=3)

plt.xticks(())

plt.yticks(())

plt.show()




