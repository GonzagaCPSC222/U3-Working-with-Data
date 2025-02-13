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
# a data science library
# built on top of NumPy
# why pandas?
# lots of great built in data science functionality
# like indexing, slicing, cleaning, stats, ...
# major shortcomings of using 2D lists of tables
# 1. lack of label-based indexing
# 2. lot of work to grab a column

# 2 main data storage objects
# 1. 1D: Series
# 2. 2D: DataFrame (every column is a Series)

# lets start with Series
# you can create a Series from many different date types
# lets make one from a list
import pandas as pd

pops = [229447, 755078, 151574, 38977]
cities = ["Spokane", "Seattle", "Bellevue", "Issaquah"]
pop_ser = pd.Series(pops, index=cities)
pop_ser.name = "Population"
print(pop_ser)

# indexing
print(pop_ser["Seattle"])
print(pop_ser[["Seattle", "Issaquah"]])
print(pop_ser["Seattle":"Issaquah"]) # label based slicing
# is inclusive of the stop label
# use .iloc[ ] for position based indexing
print(pop_ser.iloc[1])
print(pop_ser.iloc[[1, 3]])
print(pop_ser.iloc[1:3]) # position based slicing
# is exclusive of the stop label

# summary stats
print(pop_ser.mean())
print(pop_ser.std())

# we can add a new value to the series
# much like we add a new key-value pair to a dictionary
pop_ser["Sammamish"] = 65116
print(pop_ser)

# we can also make an empty Series
pop_ser2 = pd.Series(dtype=int)
pop_ser2["Federal Way"] = 97701
print(pop_ser2)
print()

# time for DataFrames!
# lets make a DataFrame from a 2D list
twod_list = [["a", 3, 99.9],
             ["b", 60, -12.1],
             ["c", 1, 5.5555]]
df = pd.DataFrame(twod_list,
                  index=["row1", "row2", "row3"],
                  columns=["col1", "col2", "col3"])
print(df)
# task: make a dataframe for our city population data
# columns: "City", "Population", "Size"
# where Size is one of "Small", "Medium", "Large"
pop_data = [["Spokane", 229447, "Large"],
            ["Seattle", 755078, "Large"],
            ["Bellevue", 151574, "Medium"],
            ["Issaquah", 38977, "Small"]]
pop_df = pd.DataFrame(pop_data, columns=["City", "Population", "Size"])
pop_df = pop_df.set_index("City")
print(pop_df)
print()

# indexing
# GS cleaning up this section a bit after class
pop_ser = pop_df["Population"]
print(pop_ser)
seattle_ser = pop_df.iloc[1]
print(seattle_ser)
seattle_pop = pop_df.iloc[1, 0]
print(seattle_pop)
# use .loc[ ] to do label based row indexing
seattle_ser = pop_df.loc["Seattle"]
print(seattle_ser)
seattle_pop = pop_df.loc["Seattle", "Population"]
print(seattle_pop)


# look into pd.read_csv("filename")
# lets load up regions.csv into a dataframe
region_df = pd.read_csv("regions.csv", index_col=0)
print(region_df)

# now lets join pop_df and region_df on "City"
# to make a 3rd DataFrame
# by default, merge() does an inner join
merged_df = pop_df.merge(region_df, on=["City"], how="outer")
print(merged_df)

# lets write the contents of merged_df to a file
# merged.csv
merged_df.to_csv("merged.csv")

# data aggregation
# gathering and presenting data in a summarized form
# lets see split apply combine in action!
# 1. split
grouped_by_size = merged_df.groupby("Size")

# short way to do # 2. apply and #3. combine
mean_pop_ser = grouped_by_size["Population"].mean()
print("short way: split apply combine results:")
print(mean_pop_ser)
print()

# GS adding after class
# longer way to do #2. apply and #3. combine
# (explaining what is going on with grouped_by_size)
print(grouped_by_size)
print(grouped_by_size.groups.keys())
large_df = grouped_by_size.get_group("Large")
print(large_df)
print(type(large_df))
# we don't want to hard code extracted each attribute value's
# data frame with get_group()
# instead, we are going to write extensible code using...
# a loop!!
mean_pop_ser = pd.Series(dtype=float)
for group_name, group_df in grouped_by_size:
    print(group_name)
    print(group_df)
    # 2. apply
    group_pop_ser = group_df["Population"]
    group_pop_mean = group_pop_ser.mean()
    print(group_pop_mean)
    # 3. combine
    mean_pop_ser[group_name] = group_pop_mean
    print("*****")

print("long way: split apply combine results:")
print(mean_pop_ser)

