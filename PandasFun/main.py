# (more on) ATTRIBUTES
# 1. what is the type
# e.g. how should it be stored?
# int, float, str, ...
# 2. what is the semantic type
# e.g. what does the attribute (and its values)
# represent?
# domain knowledge!!!
# 3. what is the measurement scale
# categorical and continuous (numeric)
# nominal: categorical without an inherent ordering
# ex: names, eye colors, occupations, zip codes, ...
# ordinal: categorical with an inherent ordering
# ex: T shirt sizes (...,S, M, L, ...)
# letter grades (A, A-, B+, ...)
# ratio-scaled: continous where 0 means absence
# ex: 0lbs means an absence of weight
# 0 degrees kelvin means absence of temperature
# interval: continous where 0 does not mean absence
# ex: 0 degrees fahrenheit does not mean absence of temp

# noisy vs invalid
# noisy: valid on the measurement scale, but
# recorded incorrectly
# ex: age attribute, someone who is 18 years old
# enter 81 years old
# invalid: not valid on the measurement scale
# ex: age attribute, someone enters "bob"

# labeled vs unlabeled (preview for our unit on machine learning)
# labeled data: if there is an attribute (called "class")
# that you are interested in predicting for "unseen"
# instances
# this is called supervised machine learning (more
# on this later...)
# if the class attribute is categorical
# then this is called a "classification task"
# if the class attribute is continuous
# then this is called a "regression task"
# unlabeled data: there is no such class attribute
# (you want to predict)
# maybe you want to use data mining to
# look for trends, groups, associations, outliers, etc...

# PANDAS
# "panel data"
# it is a library for data science
# just like numpy (and others...)
# pandas is built on top of numpy

# why pandas?
# lots of great data science functionality
# built in (like indexing, slicing, cleaning, stats,...)
# one of the major shortocomings of using lists
# for tables is the lack of label-based indexing
# grab a column by name

# there are 2 main objects for storing data
# 1D: Series
# 2D: DataFrame (where each column is a Series)

# let's start with Series
# there are several ways to make a Series
# from lists
import pandas as pd
populations = [737015, 48161, 20926, 1767]
cities = ["Seattle", "Bothell", "Mill Creek", "Ritzville"]
pop_ser = pd.Series(populations)
print(pop_ser)