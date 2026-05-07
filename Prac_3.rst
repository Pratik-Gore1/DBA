.. code:: ipython3

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

.. code:: ipython3

    df = pd.read_csv("AcademicPerformance.csv")

.. code:: ipython3

    print(df.head())
    print(df.shape)


.. parsed-literal::

       Rollno  Marks  Gender   Age  PhD
    0       1  140.0       1  47.0  Yes
    1       2   30.0       0  65.0  Yes
    2       3   35.1       0  56.0   No
    3       4   30.0       1  23.0   No
    4       5   80.0       0   NaN  Yes
    (100, 5)
    

.. code:: ipython3

    df.info()


.. parsed-literal::

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100 entries, 0 to 99
    Data columns (total 5 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   Rollno  100 non-null    int64  
     1   Marks   100 non-null    float64
     2   Gender  100 non-null    int64  
     3   Age     84 non-null     float64
     4   PhD     87 non-null     object 
    dtypes: float64(2), int64(2), object(1)
    memory usage: 4.0+ KB
    

.. code:: ipython3

    # Check Missing Values
    
    print(df.isnull().sum())


.. parsed-literal::

    Rollno     0
    Marks      0
    Gender     0
    Age       16
    PhD       13
    dtype: int64
    

.. code:: ipython3

    # Fill Missing Values
    
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].mean())
    
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

.. code:: ipython3

    print(df.isnull().sum())


.. parsed-literal::

    Rollno    0
    Marks     0
    Gender    0
    Age       0
    PhD       0
    dtype: int64
    

.. code:: ipython3

    print("Duplicate Rows :", df.duplicated().sum())


.. parsed-literal::

    Duplicate Rows : 0
    

.. code:: ipython3

    # Remove Duplicate Records
    
    df = df.drop_duplicates()

.. code:: ipython3

    # Check Inconsistencies
    
    for col in df.select_dtypes(include='object').columns:
        print(col)
        print(df[col].unique())


.. parsed-literal::

    PhD
    ['Yes' 'No']
    

.. code:: ipython3

    # Convert Text Columns To Lowercase
    
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.lower()

.. code:: ipython3

    # Display Cleaned Dataset
    
    print(df.head())


.. parsed-literal::

       Rollno  Marks  Gender        Age  PhD
    0       1  140.0       1  47.000000  yes
    1       2   30.0       0  65.000000  yes
    2       3   35.1       0  56.000000   no
    3       4   30.0       1  23.000000   no
    4       5   80.0       0  47.821429  yes
    

.. code:: ipython3

    # Select Numeric Columns
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    print(numeric_cols)


.. parsed-literal::

    Index(['Rollno', 'Marks', 'Gender', 'Age'], dtype='object')
    

.. code:: ipython3

    # Boxplot For Outlier Detection
    
    for col in numeric_cols:
        plt.figure(figsize=(5, 3))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.show()



.. image:: output_13_0.png



.. image:: output_13_1.png



.. image:: output_13_2.png



.. image:: output_13_3.png


.. code:: ipython3

    # Detect Outliers Using IQR Method
    
    for col in numeric_cols:
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
    
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
    
        outliers = df[(df[col] < lower_limit) | (df[col] > upper_limit)]
    
        print(f"{col} : {len(outliers)} outliers")
    
    print(df.shape)


.. parsed-literal::

    Rollno : 0 outliers
    Marks : 2 outliers
    Gender : 0 outliers
    Age : 0 outliers
    (100, 5)
    

.. code:: ipython3

    # Remove Outliers Using IQR Method
    
    for col in numeric_cols:
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
    
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
    
        df = df[(df[col] >= lower_limit) & (df[col] <= upper_limit)]

.. code:: ipython3

    # Dataset Shape After Removing Outliers
    
    print(df.shape)


.. parsed-literal::

    (96, 5)
    

.. code:: ipython3

    for col in numeric_cols: 
        plt.figure(figsize=(5, 3)) 
        sns.boxplot(x=df[col]) 
        plt.title(f"Boxplot of {col} After Removing Outliers") 
        plt.show()



.. image:: output_17_0.png



.. image:: output_17_1.png



.. image:: output_17_2.png



.. image:: output_17_3.png


