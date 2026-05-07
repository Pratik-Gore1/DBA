.. code:: ipython3

    import csv
    
    year_data = {}
    month_data = {}
    
    with open("weather_data.txt", "r") as file:
    
        reader = csv.reader(file)
    
        next(reader)   # Skip header
    
        for row in reader:
            try:
                date = row[0]
    
                temp = float(row[1])
                dew = float(row[2])
                wind = float(row[4])
    
                # Extract year and month
                date_part = date.split(" ")[0]
    
                month = date_part.split("/")[0]
                year = date_part.split("/")[2]
    
                # -------- YEAR WISE --------
                if year not in year_data:
                    year_data[year] = [0, 0, 0, 0]
    
                year_data[year][0] += temp
                year_data[year][1] += dew
                year_data[year][2] += wind
                year_data[year][3] += 1
    
                # -------- MONTH WISE --------
                month_key = month + "-" + year
    
                if month_key not in month_data:
                    month_data[month_key] = [0, 0, 0, 0]
    
                month_data[month_key][0] += temp
                month_data[month_key][1] += dew
                month_data[month_key][2] += wind
                month_data[month_key][3] += 1
    
            except:
                continue
    
    # -------- YEAR WISE AVERAGES --------
    print("\nYEAR WISE AVERAGES")
    
    for year in year_data:
    
        temp_avg = year_data[year][0] / year_data[year][3]
        dew_avg = year_data[year][1] / year_data[year][3]
        wind_avg = year_data[year][2] / year_data[year][3]
    
        print("\nYear:", year)
        print("Average Temperature:", round(temp_avg, 2))
        print("Average Dew Point:", round(dew_avg, 2))
        print("Average Wind Speed:", round(wind_avg, 2))
    
    # -------- MONTH WISE AVERAGES --------
    print("\nMONTH WISE AVERAGES")
    
    for month in month_data:
    
        temp_avg = month_data[month][0] / month_data[month][3]
        dew_avg = month_data[month][1] / month_data[month][3]
        wind_avg = month_data[month][2] / month_data[month][3]
    
        print("\nMonth:", month)
        print("Average Temperature:", round(temp_avg, 2))
        print("Average Dew Point:", round(dew_avg, 2))
        print("Average Wind Speed:", round(wind_avg, 2))


.. parsed-literal::

    
    YEAR WISE AVERAGES
    
    Year: 2012
    Average Temperature: 8.8
    Average Dew Point: 2.56
    Average Wind Speed: 14.95
    
    MONTH WISE AVERAGES
    
    Month: 1-2012
    Average Temperature: -7.37
    Average Dew Point: -12.29
    Average Wind Speed: 18.11
    
    Month: 2-2012
    Average Temperature: -4.23
    Average Dew Point: -9.22
    Average Wind Speed: 14.84
    
    Month: 3-2012
    Average Temperature: 3.12
    Average Dew Point: -3.49
    Average Wind Speed: 14.51
    
    Month: 4-2012
    Average Temperature: 7.01
    Average Dew Point: -1.93
    Average Wind Speed: 17.37
    
    Month: 5-2012
    Average Temperature: 16.24
    Average Dew Point: 8.08
    Average Wind Speed: 12.85
    
    Month: 6-2012
    Average Temperature: 20.13
    Average Dew Point: 11.74
    Average Wind Speed: 14.68
    
    Month: 7-2012
    Average Temperature: 22.79
    Average Dew Point: 14.6
    Average Wind Speed: 11.89
    
    Month: 8-2012
    Average Temperature: 22.28
    Average Dew Point: 15.64
    Average Wind Speed: 13.93
    
    Month: 9-2012
    Average Temperature: 16.48
    Average Dew Point: 10.76
    Average Wind Speed: 14.11
    
    Month: 10-2012
    Average Temperature: 10.95
    Average Dew Point: 6.53
    Average Wind Speed: 15.48
    
    Month: 11-2012
    Average Temperature: 0.93
    Average Dew Point: -4.18
    Average Wind Speed: 13.97
    
    Month: 12-2012
    Average Temperature: -3.31
    Average Dew Point: -6.13
    Average Wind Speed: 17.61
    

