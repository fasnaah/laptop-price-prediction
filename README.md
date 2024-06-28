# laptop-price-prediction
### Comprehensive Laptop Price Prediction Project

#### Introduction
The project aims to predict laptop prices using a dataset that includes various specifications such as company, type, RAM, CPU, GPU, and more. The analysis was conducted in three main stages: data preprocessing and exploratory data analysis (EDA) using MySQL, machine learning model development in Google Colab, and visualization using Power BI.

### Stage 1: Data Preprocessing and EDA in MySQL

#### Loading and Preparing Data
The initial step was to load the dataset into a MySQL table for cleaning and preprocessing.

**Query:**
```sql
SELECT * FROM fasna.laptop_tabl;
```
This query loads the data from the table.

**Removing 'kg' from the `Weight` column:**
```sql
UPDATE laptop_tabl SET Weight = REPLACE(Weight, 'kg', '');
ALTER TABLE laptop_tabl MODIFY COLUMN Weight FLOAT;
```
This code cleans the `Weight` column by removing 'kg' and changing its data type to `FLOAT` for numerical analysis.

#### Adding New Columns for Screen Features
New columns were added to indicate whether a laptop has specific screen features like touchscreen, IPS, and FullHD.

**Adding `Touchscreen` column:**
```sql
ALTER TABLE laptop_tabl ADD COLUMN Touchscreen varchar(30);
UPDATE laptop_tabl SET Touchscreen = 'yes' WHERE ScreenResolution LIKE '%Touchscreen%';
UPDATE laptop_tabl SET Touchscreen = 'no' WHERE ScreenResolution NOT LIKE '%Touchscreen%';
```
Similar steps were repeated for IPS and FullHD features.

#### Extracting CPU Information
The CPU information was split into more granular parts to analyze its impact on price.

**Creating `Cpu_version` and `Cpu_model`:**
```sql
ALTER TABLE laptop_tabl ADD COLUMN Cpu_version VARCHAR(255);
UPDATE laptop_tabl SET Cpu_version = CONCAT(SUBSTRING_INDEX(Cpu, ' ', 1), ' ', SUBSTRING_INDEX(SUBSTRING_INDEX(Cpu, ' ', 2), ' ', -1), ' ', SUBSTRING_INDEX(SUBSTRING_INDEX(Cpu, ' ', 3), ' ', -1));
```
These commands extract and combine specific parts of the `Cpu` field.

**Creating `Cpu_model`:**
```sql
ALTER TABLE laptop_tabl ADD Cpu_model varchar(30);
UPDATE laptop_tabl SET Cpu_model = SUBSTRING_INDEX(Cpu_version, ' ', 1);
```
This separates the CPU model from the version.

**Extracting CPU Frequency:**
```sql
UPDATE laptop_tabl SET Cpu = SUBSTRING_INDEX(Cpu, ' ', -1);
UPDATE laptop_tabl SET cpu = REPLACE(cpu, 'GHz', ' ');
ALTER TABLE laptop_tabl CHANGE cpu cpu_frequency FLOAT;
```
The CPU frequency is extracted and converted to a float.

#### Processing RAM and Memory
**Cleaning `Ram` column:**
```sql
UPDATE laptop_tabl SET Ram = REPLACE(Ram, 'GB', ' ');
ALTER TABLE laptop_tabl CHANGE Ram ram_gb INT;
```
This code cleans and converts the `Ram` column to an integer.

**Handling Memory Column:**
```sql
ALTER TABLE laptop_tabl ADD COLUMN MemoryStr VARCHAR(255);
UPDATE laptop_tabl SET MemoryStr = REPLACE(Memory, 'TB', '000GB');
ALTER TABLE laptop_tabl DROP COLUMN Memory;
```
Memory values were cleaned and converted to a more consistent format.

**Splitting and Analyzing Memory Types:**
```sql
UPDATE laptop_tabl SET MemoryStr = REPLACE(MemoryStr, 'TB', '000GB');
```
This splits the memory into different storage types (HDD, SSD, etc.).

#### Adding GPU Information
**Extracting GPU Model:**
```sql
ALTER TABLE laptop_tabl ADD Gpu_model VARCHAR(30);
UPDATE laptop_tabl SET Gpu_model = SUBSTRING_INDEX(Gpu, ' ', 1);
```
This extracts the GPU model from the `Gpu` column.

**Cleaning GPU Frequency:**
```sql
UPDATE laptop_tabl SET Gpu = TRIM(SUBSTRING(Gpu, LOCATE(' ', Gpu) + 1));
```

### Stage 2: Machine Learning Model Development in Google Colab

#### Loading Data
The dataset is loaded into a Pandas DataFrame for analysis.

```python
import pandas as pd
df = pd.read_csv('/content/laptop_data.csv')
df.head()
```

#### Exploratory Data Analysis
**Checking for Missing Values:**
```python
df.isna().sum()
```
This identifies missing values in the dataset.

**Feature Analysis:**
```python
for col in df.columns:
  print(col)
  print(df[col].nunique())
  print(df[col].value_counts())
  print('-'*100)
```
This analyzes the unique values and distribution of each column.

#### Label Encoding and Feature Selection
**Label Encoding:**
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.columns:
  df[col] = le.fit_transform(df[col])
```
This encodes categorical features into numerical values.

**Chi-Square Test for Categorical Features:**
```python
from sklearn.feature_selection import chi2
score1 = chi2(x_category, y1)
```
This selects important categorical features.

**ANOVA Test for Continuous Features:**
```python
from sklearn.feature_selection import f_classif
score2 = f_classif(x_continuous, y1)
```
This selects important continuous features.

#### Data Transformation and Visualization
**One-Hot Encoding for Categorical Features:**
```python
df = df.join(pd.get_dummies(df['Company'], dtype=int))
df = df.drop('Company', axis=1)
```
This converts categorical features into binary features.

**Visualization:**
```python
import seaborn as sns
sns.barplot(x=df['Company'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()
```
This visualizes the relationship between features and price.

#### Model Training and Evaluation
**Splitting Data:**
```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
```
This splits the data into training and test sets.

**Linear Regression:**
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
```
This trains a linear regression model.

**Random Forest Regression:**
```python
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(x_train, y_train)
y_pred = rfr.predict(x_test)
```
This trains a random forest regressor.

**Model Evaluation:**
```python
from sklearn.metrics import mean_absolute_percentage_error, r2_score
print("MAPE:", mean_absolute_percentage_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```
This evaluates the model's performance.

### Stage 3: Visualization with Power BI

The processed data and model results were visualized using Power BI, creating interactive dashboards to showcase key insights and predictions. This helps stakeholders understand the data and model performance more intuitively.

#### Conclusion
This comprehensive analysis involved data cleaning, feature engineering, exploratory data analysis, model training, and evaluation. The final model successfully predicts laptop prices based on various specifications, and the interactive dashboard provides a user-friendly way to explore the results. This project demonstrates a robust approach to predictive modeling and data visualization.
