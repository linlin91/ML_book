"""Looking into data"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get example data
housing = pd.read_csv("housing.csv")

# Look at data
housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()

housing.hist(bins=50, figsize=(20,15))
plt.show()

housing["median_income"].hist()
plt.show()

# make categories

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5) ## Divide by 1.5 to limit the number of income categories
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True) ## Label those above 5 as 5
## Or use cut when you need to segment and sort data values into bins.
## This function is also useful for going from a continuous variable to a categorical variable.
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
