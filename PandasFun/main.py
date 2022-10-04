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
pop_ser = pd.Series(populations, index=cities)
print(pop_ser)
# we can name a series
pop_ser.name = "Population"
print(pop_ser)

# indexing
print(pop_ser["Bothell"])
print(pop_ser[["Bothell", "Ritzville"]])
print(pop_ser["Bothell": "Ritzville"]) # inclusive of end when using labels
# use .iloc[] for position-based indexing
print(pop_ser.iloc[1])
print()
print(pop_ser.iloc[[1, 3]])
print()
print(pop_ser.iloc[1:3]) # exclusive of end when using position

# summary stats
print(pop_ser.mean())
print(pop_ser.std())

# we can new data to a series, much like adding
# a new key-value pair to a dictionary
pop_ser["Mukilteo"] = 21538
print(pop_ser)

# we can have an empty series
pop_ser2 = pd.Series(dtype=int)
pop_ser2["Mukilteo"] = 21538
print(pop_ser2)

# now, dataframes are used for storing 2D data using pandas
# lets make a dataframe from a 2D list
twod_list = [["a", 7, 11.1], ["b", 22, 56.3], ["c", 813, 909.99]]
column_names = ["col1", "col2", "col3"]
row_names = ["row1", "row2", "row3"]
df = pd.DataFrame(twod_list, columns=column_names, index=row_names)
print(df)
# task: create a dataframe for the population data
# 3 columns: "City", "Population", "Class"
# "Class" can be "Large" "Medium" or "Small"
pop_data = [["Seattle", 737015, "Large"],
            ["Bothell", 48161, "Medium"],
            ["Ritzville", 1767, "Small"],
            ["Spokane", 228989, "Large"]]
column_names = ["City", "Population", "Class"]
pop_df = pd.DataFrame(pop_data, columns=column_names)
pop_df = pop_df.set_index("City")
print(pop_df)

# indexing
pop_ser = pop_df["Population"]
print(pop_ser)
seattle_ser = pop_df.iloc[0]
print(seattle_ser)
seattle_pop = pop_df.iloc[0, 0]
print(seattle_pop)
# use .loc[] for label based indexing
seattle_pop = pop_df.loc["Seattle", "Population"]
print(seattle_pop)

# lets load regions.csv into a dataframe
regions_df = pd.read_csv("regions.csv", index_col=0)
print(regions_df)

# let's join pop_df with the regions_df to produce
# a third dataframe called merged_df
# we will join on the column they have in common
# ("City", which also happens to be the index)
merged_df = pop_df.merge(regions_df, on="City", how="outer")
# by default, merge does an inner join
print(merged_df)

# we can write a dataframe (and a series) to a file
merged_df.to_csv("merged.csv")

# data aggregation
# let's split merged_df on "Class" and apply the 
# mean() to all the Population series in the Class subtables
# and combine the populations means into a final Series
# 1. split
grouped_by_class = merged_df.groupby("Class")
print(grouped_by_class)
print(grouped_by_class.groups.keys())
large_df = grouped_by_class.get_group("Large")
print(large_df)
print(type(large_df))
# we don't want to hard code extracted each attribute value's
# data frame with get_group()
# instead, we are going to write extensible code using...
# a loop!!
mean_pop_ser = pd.Series(dtype=float)
for group_name, group_df in grouped_by_class:
    print(group_name)
    print(group_df)
    # 2. apply
    group_pop_ser = group_df["Population"]
    group_pop_mean = group_pop_ser.mean()
    print(group_pop_mean)
    # 3. combine
    mean_pop_ser[group_name] = group_pop_mean
    print("*****")

print("split apply combine results:")
print(mean_pop_ser)

# smaller way :)
mean_pop_ser = grouped_by_class["Population"].mean()
print(mean_pop_ser)