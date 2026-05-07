.. code:: ipython3

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

.. code:: ipython3

    iris = pd.read_csv("iris.csv")
    
    # Remove Id column
    
    if 'Id' in iris.columns:
        iris = iris.drop('Id', axis=1)
    
    iris.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>SepalLengthCm</th>
          <th>SepalWidthCm</th>
          <th>PetalLengthCm</th>
          <th>PetalWidthCm</th>
          <th>species</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>5.1</td>
          <td>3.5</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4.9</td>
          <td>3.0</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.7</td>
          <td>3.2</td>
          <td>1.3</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4.6</td>
          <td>3.1</td>
          <td>1.5</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5.0</td>
          <td>3.6</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>Iris-setosa</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    # Features and Data Types
    
    print("Features and Data Types:\n")
    print(iris.dtypes)


.. parsed-literal::

    Features and Data Types:
    
    SepalLengthCm    float64
    SepalWidthCm     float64
    PetalLengthCm    float64
    PetalWidthCm     float64
    species           object
    dtype: object
    

.. code:: ipython3

    # Subplots of Histograms
    
    features = iris.select_dtypes(include='number').columns
    
    fig, axes = plt.subplots(2, 2, figsize=(12,8))
    
    axes = axes.ravel()
    
    for i, column in enumerate(features):
    
        axes[i].hist(iris[column], bins=15)
    
        axes[i].set_title(column)
    
    plt.tight_layout()
    
    plt.show()



.. image:: output_3_0.png


.. code:: ipython3

    # Boxplot of Features
    
    plt.figure(figsize=(10,6))
    
    sns.boxplot(data=iris)
    
    plt.title("Boxplot of Iris Dataset Features")
    
    plt.show()



.. image:: output_4_0.png


.. code:: ipython3

    # Count Outliers using IQR Method
    
    numeric_columns = iris.select_dtypes(include='number').columns
    
    print("Outlier Count in Each Feature:\n")
    
    for column in numeric_columns:
    
        Q1 = iris[column].quantile(0.25)
        Q3 = iris[column].quantile(0.75)
    
        IQR = Q3 - Q1
    
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
    
        outliers = iris[
            (iris[column] < lower_limit) |
            (iris[column] > upper_limit)
        ]
    
        print(column, ":", len(outliers), "outliers")


.. parsed-literal::

    Outlier Count in Each Feature:
    
    SepalLengthCm : 0 outliers
    SepalWidthCm : 4 outliers
    PetalLengthCm : 0 outliers
    PetalWidthCm : 0 outliers
    

.. code:: ipython3

    # Remove Outliers from SepalWidthCm
    
    Q1 = iris['SepalWidthCm'].quantile(0.25)
    Q3 = iris['SepalWidthCm'].quantile(0.75)
    
    IQR = Q3 - Q1
    
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    iris_clean = iris[
        (iris['SepalWidthCm'] >= lower_limit) &
        (iris['SepalWidthCm'] <= upper_limit)
    ]
    

.. code:: ipython3

    # Boxplot Before Removing Outliers
    
    sns.boxplot(y=iris['SepalWidthCm'])
    
    plt.title("Before Removing Outliers")
    
    plt.show()



.. image:: output_7_0.png


.. code:: ipython3

    # Boxplot After Removing Outliers
    
    sns.boxplot(y=iris_clean['SepalWidthCm'])
    
    plt.title("After Removing Outliers")
    
    plt.show()



.. image:: output_8_0.png


