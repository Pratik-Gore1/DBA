.. code:: ipython3

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

.. code:: ipython3

    df = pd.read_csv("AcademicPerformance.csv")
    
    print(df.head())


.. parsed-literal::

       Rollno  Marks  Gender   Age  PhD
    0       1  140.0       1  47.0  Yes
    1       2   30.0       0  65.0  Yes
    2       3   35.1       0  56.0   No
    3       4   30.0       1  23.0   No
    4       5   80.0       0   NaN  Yes
    

.. code:: ipython3

    # Check Dataset Information
    
    print(df.info())
    
    print(df.describe())
    


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
    None
               Rollno       Marks      Gender        Age
    count  100.000000  100.000000  100.000000  84.000000
    mean    50.500000   52.524500    0.500000  47.821429
    std     29.011492   42.220933    0.502519  15.037432
    min      1.000000    0.250000    0.000000  20.000000
    25%     25.750000   20.000000    0.000000  32.750000
    50%     50.500000   39.300000    0.500000  50.000000
    75%     75.250000   75.500000    1.000000  60.250000
    max    100.000000  190.000000    1.000000  77.000000
    Rollno     0
    Marks      0
    Gender     0
    Age       16
    PhD       13
    dtype: int64
    

.. code:: ipython3

    print(df.isnull().sum())


.. parsed-literal::

    Rollno     0
    Marks      0
    Gender     0
    Age       16
    PhD       13
    dtype: int64
    

.. code:: ipython3

    # Handle Missing Values
    
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    
    df['PhD'] = df['PhD'].fillna(df['PhD'].mode()[0])
    
    print(df.isnull().sum())


.. parsed-literal::

    Rollno    0
    Marks     0
    Gender    0
    Age       0
    PhD       0
    dtype: int64
    

.. code:: ipython3

    # Check Duplicate Values
    
    print("Duplicate Rows :", df.duplicated().sum())


.. parsed-literal::

    Duplicate Rows : 0
    

.. code:: ipython3

    # Check Inconsistencies
    
    for col in df.select_dtypes(include='object').columns:
        print(col)
        print(df[col].unique())


.. parsed-literal::

    PhD
    ['Yes' 'No']
    

.. code:: ipython3

    # Convert Categorical Values To Lowercase
    
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.lower()
    
    print(df.head())


.. parsed-literal::

       Rollno  Marks  Gender        Age  PhD
    0       1  140.0       1  47.000000  yes
    1       2   30.0       0  65.000000  yes
    2       3   35.1       0  56.000000   no
    3       4   30.0       1  23.000000   no
    4       5   80.0       0  47.821429  yes
    

.. code:: ipython3

    # Find Skewness
    
    print(df.skew(numeric_only=True))


.. parsed-literal::

    Rollno    0.000000
    Marks     1.077026
    Gender    0.000000
    Age      -0.257739
    dtype: float64
    

.. code:: ipython3

    # Plot Distribution Before Transformation
    
    sns.histplot(df['Marks'], kde=True)
    
    plt.title("Marks Distribution Before Transformation")
    
    plt.show()



.. image:: output_9_0.png


.. code:: ipython3

    # Apply Square Root Transformation
    
    df['Marks'] = np.sqrt(df['Marks'])

.. code:: ipython3

    # Plot Distribution After Transformation
    
    sns.histplot(df['Marks'], kde=True)
    
    plt.title("Marks Distribution After Transformation")
    
    plt.show()



.. image:: output_11_0.png


.. code:: ipython3

    # Check Skewness After Transformation
    
    print(df['Marks'].skew())


.. parsed-literal::

    0.21202620353224017
    

.. code:: ipython3

    print(df.head())


.. parsed-literal::

       Rollno      Marks  Gender        Age  PhD
    0       1  11.832160       1  47.000000  yes
    1       2   5.477226       0  65.000000  yes
    2       3   5.924525       0  56.000000   no
    3       4   5.477226       1  23.000000   no
    4       5   8.944272       0  47.821429  yes
    


