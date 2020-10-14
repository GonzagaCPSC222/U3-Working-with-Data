# ## Data Cleaning Overview
# Data analysts spend a surprising amount of time preparing data for analysis. In fact, a survey was conducted found that cleaning big data is the most time-consuming and least enjoyable task data scientists do!
# <img src="http://blogs-images.forbes.com/gilpress/files/2016/03/Time-1200x511.jpg" width="700">
# (image from [http://blogs-images.forbes.com/gilpress/files/2016/03/Time-1200x511.jpg](http://blogs-images.forbes.com/gilpress/files/2016/03/Time-1200x511.jpg))
# 
# Data preparation includes, but is not limited to, tasks such as:
# * Loading data into an appropriate data structure
# * Merging multiple data sets
# * Cleaning the data
#     * Reshaping data, transforming data, changing data type
#     * Replacing values, removing duplicates
#     * Performing data binning/discretization
#     * Handling missing values
#     * Detecting outliers
#     * Standardizing/scaling data
# * Many others!
# 
# ### Missing Values
# It is not uncommon to have datasets with missing values. Missing values are usually coded as an out of range value, such as an empty string in a text field, -1 in a numeric field that is normally positive, or 0 in a numeric field that cannot take on the value of 0. In the Scipy ecosystem, the common value `NaN` (not a number) is used to denote missing data. There is support in the Scipy libraries to handle `NaN` specially. For example, the Pandas function [`isnull()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.isnull.html) returns a Boolean array detecting the `NaN` values element-wise and [`dropna()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html) removes `NaN` values from a series or data frame:
import numpy as np
import pandas as pd
x = np.arange(0, 10)
ser = pd.Series(x)
ser[1] = np.NaN
ser[5] = np.NaN
nans = ser.isnull()
# count the number of missing values
print(nans.sum())
print(ser)
ser.dropna(inplace=True)
print(ser)


# Note: you can learn more about missing data by reading [Pandas website](https://pandas.pydata.org/pandas-docs/stable/missing_data.html).

# By learning how to use the Pandas library, we have the skills to perform many of the tasks listed above. In this lesson, we are going to focus on *data cleaning*, modifying the data to make it sufficiently accurate and structured to support the analysis you want to perform. To learn about data cleaning, we are going to clean data by working through an example!

# ## Data Cleaning Example
# We are going to work with the [pd_hoa_activities.csv](https://raw.githubusercontent.com/gsprint23/aha/master/lessons/files/pd_hoa_activities.csv) dataset. This dataset contains information from a smart home study where participants performed 9 activities of daily living (ADLs) in a smart home environment:
# 1. Water plants
# 1. Fill medication dispenser
# 1. Wash counter top
# 1. Sweep and dust
# 1. Cook
# 1. Wash hands
# 1. Perform the [Timed Up and Go (TUG)](http://www.rehabmeasures.org/Lists/RehabMeasures/DispForm.aspx?ID=903) test
# 1. Perform TUG with questions being asked
# 1. A day out task
# 
# Note: you can read more about the design of this study and the various tasks in [Cook et al., 2015](http://ieeexplore.ieee.org/document/7181652/). 
# 
# The activities were timed and the duration is recorded for each participant in the dataset. The participants of the study include individual's with Parkinson's disease (PD) and age-matched, healthy older adults (HOA). For each participant in the study, the dataset includes a participant id (pid), age, and their class (PD or HOA). The data has been de-identified. For the purposes of our analysis today, we are interested in aggregating this data into PD and HOA groups to investigate the effect of PD on older adult's ability to perform the above tasks.
# 
# Here is a sample of the format of the data:
# 
# |pid|task|duration|age|class|
# |-|-|-|-|-|
# |0|1|146|72|HOA|
# |0|2|210|72|HOA|
# |0|3|241|72|HOA|
# |0|4|328|72|HOA|
# |0|5|229|72|HOA|
# |0|6|38|72|HOA|
# |0|7|10|72|HOA|
# |0|8|10|72|HOA|
# |0|dot|680|72|HOA|
# |1|1|63|54|HOA|
# |...|...|...|...|...|
# 
# Let's take a look at each column in the data and how the data needs to be cleaned:
# * pid (integer): Index of the dataset. Counting numbers starting at 0.
# * task (integer): ID of the task the patient performed.
#     * Clean: Decode the integer task label to the plain text string task label.
#     * Example: 1 will be decoded to "Water plants".
# * duration (integer): Number of seconds it took the participant to perform the task.
#     * Clean: Ensure this data is a numeric data type.
# * age (integer): Age of the participant.
#     * Clean: Ensure this data is a numeric data type.
# * class (string): Population the participant belongs to: HOA or PD.
# 
# ### Load the Data
# First we are going to load the data into a `pandas` `DataFrame` object. The header row is the first row in the file. We are not going to set an index column for the data because there is not a column in the csv file that contains unique values.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fname = "pd_hoa_activities.csv"
df = pd.read_csv(fname, header=0)
print(df.shape)
print("Number of participants:", df.shape[0] // 9)


# ### Explore the Data
# Now, let's take a look at some of the data points.
print(df.head(n=5))
print(df.tail(n=5))
print(df[660:670])
print(df[7:10])
print(df[25:28])


# If we only look at the first 5 rows and the last 5 rows of the dataset, the columns looks like it is well formed with no missing values; however, we see the class column has inconsistent labels for our two classes (HOA and PD) and for pids 663, 664, 665 (among others) there is a "?" denoting a missing value. In fact, if we count the number of "?" in the duration column, we see that there are 10 tasks with missing durations:
print(df["duration"].value_counts()["?"])


fname = "pd_hoa_activities.csv"
df = pd.read_csv(fname, header=0)


# ### Missing Data
# Let's replace the "?" with `NaN` values so we can more easily detect the fields with missing data:

df.replace("?", np.NaN, inplace=True)


# Now let's look at the data in each column.
for col in df.columns:
    ser = df[col]
    print(ser.value_counts())
    print("Number of NaN:", ser.isnull().sum())
    print("***********************************************************************\n")


# Based on our exploration of the data, we know there are 10 null values in the duration column that we need handle. There are a few ways we do this:
# 1. Remove the row and/or participant's data with missing information
# 1. Fill the missing values. One way to do this is by filling each missing value with the average of "similar" instances (e.g. same task, same class, similar age).
# 1. Leave it alone for now. Handle it on a case by case basis in the later stages of the data analysis pipeline.
# 
# We are going to remove the rows with missing information.

print("Before cleaning:", df.shape)  
df.dropna(inplace=True)
index = np.arange(0, len(df))
df.set_index(index, inplace=True)
print("After cleaning:", df.shape)  


# ### Decode Task
# Now, let's decode the integer values associated with the task column by replacing them with a more human-readable text label. We will use a dictionary to story the integer to string mappings for task codes and replace the integers with the strings in place.

task_decoder = {"1": "Water Plants", "2": "Fill Medication Dispenser", "3": "Wash Countertop",                "4": "Sweep and Dust", "5": "Cook", "6": "Wash Hands", "7": "Perform TUG",                "8": "Perform TUG w/Questions", "dot": "Day Out Task"}

def decode_task(df):
    '''
    
    '''
    ser = df["task"]
    for key in task_decoder:
        ser.replace(key, task_decoder[key], inplace=True)
decode_task(df)
print(df.head(n=11))


# Looking at our data frame now, we see that the task category is much more readable. This will be especially useful for generating plots with task labels.

# ### Check Numeric Data Types
# Now, let's check out the data types for our two numeric columns, duration and age:

print(df["duration"].dtype)
print(df["age"].dtype)


# We see that the age column is an integer type, but the duration column is an object type. This means that Pandas was unable to infer this column contained all integers when it was read in, which makes sense since we know there were "?"s in the duration column. Since we have replaced the "?" with `NaN`, let's convert it to integer now:

df["duration"] = df["duration"].astype(np.int)
print(df["duration"].dtype)


# ### Clean Class
# Lastly, we are going to clean the class column. This column is quite messy compared to the other columns we have worked with. We will use a simple rule based system to handle the various spellings and word choices that represent "HOA" and "PD".
# 
# Note: If there are entries that we cannot classify as one of the above labels, we will overwrite the entry with a null value (`NaN`) to represent missing data.

def clean_class(df):
    '''
    
    '''
    ser = df["class"].copy()
    
    for i in range(0, len(ser), 1):
        curr = str(ser[i])
        curr = curr.lower()
        if "hoa" in curr or "healthy" in curr:
            ser[i] = "HOA"
        elif "pd" in curr or "parkinson" in curr:
            ser[i] = "PD"
        else:
            print("Unrecognized status: %d, %s" %(i, ser[i]))
            ser[i] = np.NaN
        
    df["class"] = ser

clean_class(df)
print(df.head())
print(df["class"].value_counts())


# Now, we will write the cleaned data frame out to a new file. Our dataset is now cleaned and ready for use in the next step of our data analysis pipeline. Depending on what we want to do with the data, this could be continuing exploration by generating visualizations of the data, or perhaps scaling the features in preparation for machine learning.


out_fname = "pd_hoa_activities_cleaned.csv"
df.to_csv(out_fname, index=False) # don't write out the index column

