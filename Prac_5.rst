.. code:: ipython3

    import pandas as pd
    import numpy as np
    import seaborn as sns

.. code:: ipython3

    titanic = sns.load_dataset('titanic')
    
    print(titanic.head())


.. parsed-literal::

       survived  pclass     sex   age  sibsp  parch     fare embarked  class  \
    0         0       3    male  22.0      1      0   7.2500        S  Third   
    1         1       1  female  38.0      1      0  71.2833        C  First   
    2         1       3  female  26.0      0      0   7.9250        S  Third   
    3         1       1  female  35.0      1      0  53.1000        S  First   
    4         0       3    male  35.0      0      0   8.0500        S  Third   
    
         who  adult_male deck  embark_town alive  alone  
    0    man        True  NaN  Southampton    no  False  
    1  woman       False    C    Cherbourg   yes  False  
    2  woman       False  NaN  Southampton   yes   True  
    3  woman       False    C  Southampton   yes  False  
    4    man        True  NaN  Southampton    no   True  
    

.. code:: ipython3

    print(titanic.info())


.. parsed-literal::

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 15 columns):
     #   Column       Non-Null Count  Dtype   
    ---  ------       --------------  -----   
     0   survived     891 non-null    int64   
     1   pclass       891 non-null    int64   
     2   sex          891 non-null    object  
     3   age          714 non-null    float64 
     4   sibsp        891 non-null    int64   
     5   parch        891 non-null    int64   
     6   fare         891 non-null    float64 
     7   embarked     889 non-null    object  
     8   class        891 non-null    category
     9   who          891 non-null    object  
     10  adult_male   891 non-null    bool    
     11  deck         203 non-null    category
     12  embark_town  889 non-null    object  
     13  alive        891 non-null    object  
     14  alone        891 non-null    bool    
    dtypes: bool(2), category(2), float64(2), int64(4), object(5)
    memory usage: 80.7+ KB
    None
    

.. code:: ipython3

    # Summary Statistics of Numeric Variables Grouped by Sex and Age
    
    grouped_stats = titanic.groupby('sex')['age'].agg(['mean', 'median', 'min', 'max', 'std'])
    
    print(grouped_stats)


.. parsed-literal::

                 mean  median   min   max        std
    sex                                             
    female  27.915709    27.0  0.75  63.0  14.110146
    male    30.726645    29.0  0.42  80.0  14.678201
    

.. code:: ipython3

    # Create Numeric Values for Categorical Variable
    
    titanic['sex_numeric'] = titanic['sex'].map({
        'male': 0,
        'female': 1
    })
    
    print(titanic[['sex', 'sex_numeric']].head())


.. parsed-literal::

          sex  sex_numeric
    0    male            0
    1  female            1
    2  female            1
    3  female            1
    4    male            0
    

.. code:: ipython3

    iris = sns.load_dataset('iris')
    
    print(iris.head())


.. parsed-literal::

       sepal_length  sepal_width  petal_length  petal_width species
    0           5.1          3.5           1.4          0.2  setosa
    1           4.9          3.0           1.4          0.2  setosa
    2           4.7          3.2           1.3          0.2  setosa
    3           4.6          3.1           1.5          0.2  setosa
    4           5.0          3.6           1.4          0.2  setosa
    

.. code:: ipython3

    # Display Statistical Details for Iris-setosa
    
    setosa = iris[iris['species'] == 'setosa']
    
    print("Statistics for Iris-setosa")
    print(setosa.describe())


.. parsed-literal::

    Statistics for Iris-setosa
           sepal_length  sepal_width  petal_length  petal_width
    count      50.00000    50.000000     50.000000    50.000000
    mean        5.00600     3.428000      1.462000     0.246000
    std         0.35249     0.379064      0.173664     0.105386
    min         4.30000     2.300000      1.000000     0.100000
    25%         4.80000     3.200000      1.400000     0.200000
    50%         5.00000     3.400000      1.500000     0.200000
    75%         5.20000     3.675000      1.575000     0.300000
    max         5.80000     4.400000      1.900000     0.600000
    

.. code:: ipython3

    # Display Statistical Details for Iris-versicolor
    
    versicolor = iris[iris['species'] == 'versicolor']
    
    print("Statistics for Iris-versicolor")
    print(versicolor.describe())


.. parsed-literal::

    Statistics for Iris-versicolor
           sepal_length  sepal_width  petal_length  petal_width
    count     50.000000    50.000000     50.000000    50.000000
    mean       5.936000     2.770000      4.260000     1.326000
    std        0.516171     0.313798      0.469911     0.197753
    min        4.900000     2.000000      3.000000     1.000000
    25%        5.600000     2.525000      4.000000     1.200000
    50%        5.900000     2.800000      4.350000     1.300000
    75%        6.300000     3.000000      4.600000     1.500000
    max        7.000000     3.400000      5.100000     1.800000
    

.. code:: ipython3

    # Display Statistical Details for Iris-virginica
    
    virginica = iris[iris['species'] == 'virginica']
    
    print("Statistics for Iris-virginica")
    print(virginica.describe())


.. parsed-literal::

    Statistics for Iris-virginica
           sepal_length  sepal_width  petal_length  petal_width
    count      50.00000    50.000000     50.000000     50.00000
    mean        6.58800     2.974000      5.552000      2.02600
    std         0.63588     0.322497      0.551895      0.27465
    min         4.90000     2.200000      4.500000      1.40000
    25%         6.22500     2.800000      5.100000      1.80000
    50%         6.50000     3.000000      5.550000      2.00000
    75%         6.90000     3.175000      5.875000      2.30000
    max         7.90000     3.800000      6.900000      2.50000
    

.. code:: ipython3

    # Display Percentiles for Each Species
    
    print("Percentiles for Iris-setosa")
    print(setosa.select_dtypes(include='number').quantile([0.25, 0.50, 0.75]))
    
    print("\nPercentiles for Iris-versicolor")
    print(versicolor.select_dtypes(include='number').quantile([0.25, 0.50, 0.75]))
    
    print("\nPercentiles for Iris-virginica")
    print(virginica.select_dtypes(include='number').quantile([0.25, 0.50, 0.75]))


.. parsed-literal::

    Percentiles for Iris-setosa
          sepal_length  sepal_width  petal_length  petal_width
    0.25           4.8        3.200         1.400          0.2
    0.50           5.0        3.400         1.500          0.2
    0.75           5.2        3.675         1.575          0.3
    
    Percentiles for Iris-versicolor
          sepal_length  sepal_width  petal_length  petal_width
    0.25           5.6        2.525          4.00          1.2
    0.50           5.9        2.800          4.35          1.3
    0.75           6.3        3.000          4.60          1.5
    
    Percentiles for Iris-virginica
          sepal_length  sepal_width  petal_length  petal_width
    0.25         6.225        2.800         5.100          1.8
    0.50         6.500        3.000         5.550          2.0
    0.75         6.900        3.175         5.875          2.3
    

