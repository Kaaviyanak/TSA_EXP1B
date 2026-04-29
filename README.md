# Ex.No: 1B                     CONVERSION OF NON STATIONARY TO STATIONARY DATA
# Date:25/04/2026 
# NAME : Kaaviyan K
# REG : 212224240066

### AIM:
To perform regular differncing,seasonal adjustment and log transformatio on international airline passenger data
### ALGORITHM:
1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the data preprocessing if needed and apply regular differncing,seasonal adjustment,log transformation.
4. Plot the data according to need, before and after regular differncing,seasonal adjustment,log transformation.
5. Display the overall results.
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


data = pd.read_csv("/content/user_behavior_timeseries.csv")

data.columns = data.columns.str.strip()


data['Date'] = pd.date_range(start='2024-01-01', periods=len(data))
data['Date'] = pd.to_datetime(data['Date'])


data.set_index('Date', inplace=True)


col = "App Usage Time"

data['usage_diff'] = data[col] - data[col].shift(1)


result = seasonal_decompose(data[col], model='additive', period=7)
data['usage_sea_diff'] = result.resid

data['usage_log'] = np.log(data[col])


data['usage_log_diff'] = data['usage_log'] - data['usage_log'].shift(1)


result = seasonal_decompose(data['usage_log_diff'].dropna(), model='additive', period=7)
data['usage_log_seasonal_diff'] = result.resid


plt.figure(figsize=(16, 16))

plt.subplot(6,1,1)
plt.plot(data[col])
plt.title("Original Data")

plt.subplot(6,1,2)
plt.plot(data['usage_diff'])
plt.title("Regular Differencing")


plt.subplot(6,1,3)
plt.plot(data['usage_sea_diff'])
plt.title("Seasonal Adjustment")

plt.subplot(6,1,4)
plt.plot(data['usage_log'])
plt.title("Log Transformation")

plt.subplot(6,1,5)
plt.plot(data['usage_log_diff'])
plt.title("Log + Differencing")

plt.subplot(6,1,6)
plt.plot(data['usage_log_seasonal_diff'])
plt.title("Log + Seasonal Differencing")

plt.tight_layout()
plt.show()
```



### OUTPUT:
<img width="744" height="739" alt="image" src="https://github.com/user-attachments/assets/5e760958-09e5-49a3-b661-24ce257670c8" />


### RESULT:
Thus we have created the python code for the conversion of non stationary to stationary data on international airline passenger
data.
