.. code:: ipython3

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    dataset = sns.load_dataset('titanic')
    
    print(dataset.head())


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

    # Plot Histogram of Fare using Matplotlib
    plt.hist(dataset['fare'], bins=30)
    
    plt.title('Histogram of Fare')
    plt.xlabel('Fare')
    plt.ylabel('Frequency')
    
    plt.show()



.. image:: output_1_0.png


.. code:: ipython3

    # Plot Histplot of Fare using Seaborn
    sns.histplot(dataset['fare'], bins=30, kde=True)
    
    plt.title('Histplot of Fare')
    plt.xlabel('Fare')
    plt.ylabel('Count')
    plt.show()



.. image:: output_2_0.png


.. code:: ipython3

    # Plot Distplot of Fare
    sns.distplot(dataset['fare'])
    
    plt.title('Distplot of Fare')
    
    plt.show()


.. parsed-literal::

    C:\Users\prati\AppData\Local\Temp\ipykernel_15304\4150994087.py:2: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(dataset['fare'])
    


.. image:: output_3_1.png


.. code:: ipython3

    # Plot Distplot without KDE
    
    sns.distplot(dataset['fare'], kde=False, bins=5)
    
    plt.title('Distplot without KDE')
    
    plt.show()


.. parsed-literal::

    C:\Users\prati\AppData\Local\Temp\ipykernel_15304\4098687804.py:3: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(dataset['fare'], kde=False, bins=5)
    


.. image:: output_4_1.png


.. code:: ipython3

    # Plot Distplot without Histogram
    sns.distplot(dataset['fare'], hist=False)
    
    plt.title('KDE Plot of Fare')
    
    plt.show()


.. parsed-literal::

    C:\Users\prati\AppData\Local\Temp\ipykernel_15304\2906372147.py:2: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(dataset['fare'], hist=False)
    


.. image:: output_5_1.png


.. code:: ipython3

    # Plot Jointplot between Age and Fare
    sns.jointplot(x='age', y='fare', data=dataset)
    
    plt.show()



.. image:: output_6_0.png


.. code:: ipython3

    #  Plot Rug Plot of Fare
    sns.rugplot(x=dataset['fare'])
    
    plt.title('Rug Plot of Fare')
    
    plt.show()



.. image:: output_7_0.png


.. code:: ipython3

    
    sns.barplot(x='class', y='fare', data=dataset)
    
    plt.title('Barplot of Fare vs Class')
    
    plt.show()



.. image:: output_8_0.png




