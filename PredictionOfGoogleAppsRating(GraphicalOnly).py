# # GOOGLE APP'S RATINGS
# In[1]:
# Importing the modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')

# In[2]:
# Reading data from file
google_data = pd.read_csv('googleplaystore.csv')

# In[3]:
# Inspecting the first 5 rows
google_data.head()

# In[4]:
google_data.shape

# In[5]:
# Summary Statistics
google_data.describe()

# In[6]:
google_data.boxplot()

# In[7]:
google_data.hist()

# In[8]:
google_data.info()

# ### Data Cleaning
# In[9]:
# Count the number of missing values in the Dataframe
google_data.isnull().any()

# In[10]:
# Count the number of missing values in each column
google_data.isnull().sum()

# In[11]:
# Check how many ratings are more than 5 - Outliers
google_data[google_data.Rating > 5]

# In[12]:
google_data.drop([10472], inplace=True)

# In[13]:
google_data[10470:10475]

# In[14]:
google_data.boxplot()

# In[15]:
google_data.hist()

# ### Remove columns that are 90% empty
# In[16]:
threshold = len(google_data) * 0.1
threshold

# In[17]:
google_data.dropna(thresh=threshold, axis=1, inplace=True)

# In[18]:
print(google_data.isnull().sum())


# ### Data Manipulation
# In[19]:
# Define a function impute_median to fill the null values with appropriate
# values using aggregate functions such as mean, median or mode
def impute_median(series):
    return series.fillna(series.median())

# In[20]:
google_data.Rating = google_data['Rating'].transform(impute_median)

# In[21]:
# Count the number of null values in each column
google_data.isnull().sum()

# In[22]:
# modes of categorical values
print(google_data['Type'].mode())
print(google_data['Current Ver'].mode())
print(google_data['Android Ver'].mode())

# In[23]:
# Fill the missing categorical values with mode
google_data['Type'].fillna(str(google_data['Type'].mode().values[0]), inplace=True)
google_data['Current Ver'].fillna(str(google_data['Current Ver'].mode().values[0]), inplace=True)
google_data['Android Ver'].fillna(str(google_data['Android Ver'].mode().values[0]), inplace=True)

# In[24]:
# Count the number of null values in each column
google_data.isnull().sum()

# In[25]:
# Let's convert Price, Reviews and Ratings into Numerical Values
google_data['Price'] = google_data['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
google_data['Price'] = google_data['Price'].apply(lambda x: float(x))
google_data['Reviews'] = pd.to_numeric(google_data['Reviews'], errors='coerce')

# In[26]:
google_data['Installs'] = google_data['Installs'].apply(lambda x: str(x).replace('+', '') if '+' in str(x) else str(x))
google_data['Installs'] = google_data['Installs'].apply(lambda x: str(x).replace(',', '') if ',' in str(x) else str(x))
google_data['Installs'] = google_data['Installs'].apply(lambda x: float(x))

# In[27]:
google_data.head(10)

# In[28]:
google_data.describe()

# ### Data Visualization
# In[29]:
grp = google_data.groupby('Category')
x = grp['Rating'].agg(np.mean)
y = grp['Price'].agg(np.sum)
z = grp['Reviews'].agg(np.mean)
print(x)
print(y)
print(z)

# In[30]:
plt.figure(figsize=(12, 5))
plt.plot(x, "ro", color='g')
plt.xticks(rotation=90)
plt.show()

# In[31]:
plt.figure(figsize=(16, 5))
plt.plot(x, 'ro', color='r')
plt.xticks(rotation=90)
plt.title('Category wise Rating')
plt.xlabel('Categories-->')
plt.ylabel('Ratings-->')
plt.show()

# In[32]:
plt.figure(figsize=(16, 5))
plt.plot(y, 'r--', color='b')
plt.xticks(rotation=90)
plt.title('Category wise Pricing')
plt.xlabel('Categories-->')
plt.ylabel('Prices-->')
plt.show()

# In[33]:
plt.figure(figsize=(16, 5))
plt.plot(z, 'bs', color='g')
plt.xticks(rotation=90)
plt.title('Category wise Reviews')
plt.xlabel('Categories-->')
plt.ylabel('Reviews-->')
plt.show()
