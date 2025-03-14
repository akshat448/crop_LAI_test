import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "/Users/akshat/Developer/ML_WORK/pycrop-yield-prediction/data/dataset.csv"
data = pd.read_csv(file_path)

# Check for rows with -1 or 0 values in 'RICE YIELD (Kg per ha)' and 'WHEAT YIELD (Kg per ha)'
rice_yield_invalid = data[(data['RICE YIELD (Kg per ha)'] == -1) | (data['RICE YIELD (Kg per ha)'] == 0)]
wheat_yield_invalid = data[(data['WHEAT YIELD (Kg per ha)'] == -1) | (data['WHEAT YIELD (Kg per ha)'] == 0)]

# Print the number of rows with -1 or 0 values
print(f"Number of rows with -1 or 0 in 'RICE YIELD (Kg per ha)': {len(rice_yield_invalid)}")
print(f"Number of rows with -1 or 0 in 'WHEAT YIELD (Kg per ha)': {len(wheat_yield_invalid)}")

# Remove rows with -1 or 0 values in 'RICE YIELD (Kg per ha)' and 'WHEAT YIELD (Kg per ha)'
data_cleaned = data[(data['RICE YIELD (Kg per ha)'] != -1) & (data['RICE YIELD (Kg per ha)'] != 0) & 
                    (data['WHEAT YIELD (Kg per ha)'] != -1) & (data['WHEAT YIELD (Kg per ha)'] != 0)]

# Define the conversion function
def convert_kg_per_ha_to_bu_per_acre(yield_data):
    """
    Convert yield data from kg/ha to bushels per acre.
    Note: 1 kg/ha = 0.014867 bushels per acre for soybeans.
    """
    conversion_factor = 0.014867
    yield_data_bu_per_acre = yield_data * conversion_factor
    return yield_data_bu_per_acre

# Apply the conversion function to the yield data columns
data_cleaned['RICE YIELD (bu per acre)'] = convert_kg_per_ha_to_bu_per_acre(data_cleaned['RICE YIELD (Kg per ha)'])
data_cleaned['WHEAT YIELD (bu per acre)'] = convert_kg_per_ha_to_bu_per_acre(data_cleaned['WHEAT YIELD (Kg per ha)'])


# Basic statistics
print("\nBasic Statistics:")
print(data_cleaned.describe())

# Plot histograms for 'RICE YIELD (Kg per ha)' and 'WHEAT YIELD (Kg per ha)'
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
data_cleaned['RICE YIELD (Kg per ha)'].hist(bins=50)
plt.title('RICE YIELD (Kg per ha)')
plt.xlabel('Yield (Kg per ha)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
data_cleaned['WHEAT YIELD (Kg per ha)'].hist(bins=50)
plt.title('WHEAT YIELD (Kg per ha)')
plt.xlabel('Yield (Kg per ha)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Function to calculate the number of outliers
def count_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return len(outliers)

# Count and print the number of outliers
rice_yield_outliers = count_outliers(data_cleaned['RICE YIELD (Kg per ha)'])
wheat_yield_outliers = count_outliers(data_cleaned['WHEAT YIELD (Kg per ha)'])

print(f"Number of outliers in 'RICE YIELD (Kg per ha)': {rice_yield_outliers}")
print(f"Number of outliers in 'WHEAT YIELD (Kg per ha)': {wheat_yield_outliers}")

# Plot boxplots for 'RICE YIELD (Kg per ha)' and 'WHEAT YIELD (Kg per ha)'
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
data_cleaned.boxplot(column='RICE YIELD (Kg per ha)')
plt.title('RICE YIELD (Kg per ha)')

plt.subplot(1, 2, 2)
data_cleaned.boxplot(column='WHEAT YIELD (Kg per ha)')
plt.title('WHEAT YIELD (Kg per ha)')

plt.tight_layout()
plt.show()

# Plot yield data over the years
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
data_cleaned.groupby('Year')['RICE YIELD (Kg per ha)'].mean().plot()
plt.title('Average RICE YIELD (Kg per ha) Over Years')
plt.xlabel('Year')
plt.ylabel('Average Yield (Kg per ha)')

plt.subplot(1, 2, 2)
data_cleaned.groupby('Year')['WHEAT YIELD (Kg per ha)'].mean().plot()
plt.title('Average WHEAT YIELD (Kg per ha) Over Years')
plt.xlabel('Year')
plt.ylabel('Average Yield (Kg per ha)')

plt.tight_layout()
plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the dataset with appropriate column names
# df = pd.read_csv('/Users/akshat/Developer/ML_WORK/pycrop-yield-prediction/data/Yield Final High Quality.csv', header=None, names=['Year', 'State_Code', 'Some_Number', 'Yield'])

# # Clean the dataset by removing rows with -1 or 0 values in 'Yield'
# df_cleaned = df[df['Yield'] > 0]

# # Print basic statistics of the cleaned data
# print("Basic Statistics of the Cleaned Data:")
# print(df_cleaned.describe())

# # Plot histograms for 'Yield'
# plt.figure(figsize=(12, 6))

# sns.histplot(df_cleaned['Yield'], kde=True)
# plt.title('Histogram of Yield')

# plt.tight_layout()
# plt.show()

# # Function to count outliers
# def count_outliers(series):
#     Q1 = series.quantile(0.25)
#     Q3 = series.quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outliers = series[(series < lower_bound) | (series > upper_bound)]
#     return len(outliers)

# # Count and print the number of outliers
# yield_outliers = count_outliers(df_cleaned['Yield'])

# print(f"Number of outliers in Yield: {yield_outliers}")

# # Plot boxplot for 'Yield'
# plt.figure(figsize=(12, 6))

# sns.boxplot(y=df_cleaned['Yield'])
# plt.title('Boxplot of Yield')

# plt.tight_layout()
# plt.show()

# # Plot yield data over the years
# plt.figure(figsize=(12, 6))

# df_grouped = df_cleaned.groupby('Year').mean().reset_index()

# plt.plot(df_grouped['Year'], df_grouped['Yield'], label='Yield')

# plt.xlabel('Year')
# plt.ylabel('Yield')
# plt.title('Average Yield Data Over the Years')
# plt.legend()
# plt.show()