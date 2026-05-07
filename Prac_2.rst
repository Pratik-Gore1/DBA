.. code:: ipython3

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

.. code:: ipython3

    # Dataset Information
    
    print("Dataset Name : Titanic Dataset")
    print("Source URL : https://www.kaggle.com/datasets/yasserh/titanic-dataset")
    print("Description : The dataset contains information about passengers such as age, gender, ticket fare, class, and survival status.")


.. parsed-literal::

    Dataset Name : Titanic Dataset
    Source URL : https://www.kaggle.com/datasets/yasserh/titanic-dataset
    Description : The dataset contains information about passengers such as age, gender, ticket fare, class, and survival status.
    

.. code:: ipython3

    df = pd.read_csv("data.csv")
    
    print(df.head())


.. parsed-literal::

       PassengerId  Survived  Pclass  \
    0            1         0       3   
    1            2         1       1   
    2            3         1       3   
    3            4         1       1   
    4            5         0       3   
    
                                                    Name     Sex   Age  SibSp  \
    0                            Braund, Mr. Owen Harris    male  22.0      1   
    1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
    2                             Heikkinen, Miss. Laina  female  26.0      0   
    3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
    4                           Allen, Mr. William Henry    male  35.0      0   
    
       Parch            Ticket     Fare Cabin Embarked  
    0      0         A/5 21171   7.2500   NaN        S  
    1      0          PC 17599  71.2833   C85        C  
    2      0  STON/O2. 3101282   7.9250   NaN        S  
    3      0            113803  53.1000  C123        S  
    4      0            373450   8.0500   NaN        S  
    

.. code:: ipython3

    # Check Dimensions
    
    print("Number of Rows and Columns :")
    print(df.shape)


.. parsed-literal::

    Number of Rows and Columns :
    (891, 12)
    

.. code:: ipython3

    print(df.info())


.. parsed-literal::

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    None
    

.. code:: ipython3

    print(df.isnull().sum())


.. parsed-literal::

    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64
    

.. code:: ipython3

    df['Age'] = df['Age'].fillna(df['Age'].mean())
    
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    df = df.drop('Cabin', axis=1)
    
    print(df.isnull().sum())


.. parsed-literal::

    PassengerId    0
    Survived       0
    Pclass         0
    Name           0
    Sex            0
    Age            0
    SibSp          0
    Parch          0
    Ticket         0
    Fare           0
    Embarked       0
    dtype: int64
    

.. code:: ipython3

    # Statistical Summary
    print(df.describe())


.. parsed-literal::

           PassengerId    Survived      Pclass         Age       SibSp  \
    count   891.000000  891.000000  891.000000  891.000000  891.000000   
    mean    446.000000    0.383838    2.308642   29.699118    0.523008   
    std     257.353842    0.486592    0.836071   13.002015    1.102743   
    min       1.000000    0.000000    1.000000    0.420000    0.000000   
    25%     223.500000    0.000000    2.000000   22.000000    0.000000   
    50%     446.000000    0.000000    3.000000   29.699118    0.000000   
    75%     668.500000    1.000000    3.000000   35.000000    1.000000   
    max     891.000000    1.000000    3.000000   80.000000    8.000000   
    
                Parch        Fare  
    count  891.000000  891.000000  
    mean     0.381594   32.204208  
    std      0.806057   49.693429  
    min      0.000000    0.000000  
    25%      0.000000    7.910400  
    50%      0.000000   14.454200  
    75%      0.000000   31.000000  
    max      6.000000  512.329200  
    

.. code:: ipython3

    print(df.dtypes)


.. parsed-literal::

    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Embarked        object
    dtype: object
    

.. code:: ipython3

    # Data Type Conversion
    
    df['Survived'] = df['Survived'].astype('category')
    
    df['Pclass'] = df['Pclass'].astype('category')
    
    print(df.dtypes)


.. parsed-literal::

    PassengerId       int64
    Survived       category
    Pclass         category
    Name             object
    Sex              object
    Age             float64
    SibSp             int64
    Parch             int64
    Ticket           object
    Fare            float64
    Embarked         object
    dtype: object
    

.. code:: ipython3

    # Data Normalization
    
    scaler = MinMaxScaler()
    
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
    
    print(df[['Age', 'Fare']].head())


.. parsed-literal::

            Age      Fare
    0  0.271174  0.014151
    1  0.472229  0.139136
    2  0.321438  0.015469
    3  0.434531  0.103644
    4  0.434531  0.015713
    

.. code:: ipython3

    print(df.head())


.. parsed-literal::

       PassengerId Survived Pclass  \
    0            1        0      3   
    1            2        1      1   
    2            3        1      3   
    3            4        1      1   
    4            5        0      3   
    
                                                    Name     Sex       Age  SibSp  \
    0                            Braund, Mr. Owen Harris    male  0.271174      1   
    1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  0.472229      1   
    2                             Heikkinen, Miss. Laina  female  0.321438      0   
    3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  0.434531      1   
    4                           Allen, Mr. William Henry    male  0.434531      0   
    
       Parch            Ticket      Fare Embarked  
    0      0         A/5 21171  0.014151        S  
    1      0          PC 17599  0.139136        C  
    2      0  STON/O2. 3101282  0.015469        S  
    3      0            113803  0.103644        S  
    4      0            373450  0.015713        S  
    

