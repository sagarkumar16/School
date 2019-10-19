
# coding: utf-8



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np






mag_slow = pd.read_csv('magnet_slow.csv')

###### Creating an envelope for the Signal #######

#Getting peaks by searching for points where V(t-1) < V(t) > V(t+1)
peaks = []
for i in range(1, len(mag_slow)-1):
    if mag_slow.Voltage[i-1] < mag_slow.Voltage[i] and mag_slow.Voltage[i+1] < mag_slow.Voltage[i]:
        peaks.append((mag_slow.Time[i], mag_slow.Voltage[i]))


# voltage saturates just after t = 20s
peak_times = np.array([t[0] for t in peaks if t[0] < 20])
peak_v = np.array([v[1] for v in peaks if v[0] < 20])


#linear fit for the top part of the envelope (m = slope, b = intercept)
m,b = np.polyfit(peak_times, peak_v, 1)


#Getting troughs by searching for points where V(t-1) > V(t) < V(t+1)
troughs = []
for i in range(1, len(mag_slow)-1):
    if mag_slow.Voltage[i-1] > mag_slow.Voltage[i] and mag_slow.Voltage[i+1] > mag_slow.Voltage[i]:
        troughs.append((mag_slow.Time[i], mag_slow.Voltage[i]))

#filtering out the saturated bit again
trough_times = np.array([t[0] for t in troughs if t[0] < 20])
trough_v = np.array([v[1] for v in troughs if v[0] < 20])


#linear fit for the bottom part of the envelope (n = slope, c = intercept)
n, c = np.polyfit(trough_times, trough_v, 1)

# Getting the perpendicular distance between the two lines using basic trigonometry
d = (b-c)* math.cos(math.atan(n))

# d ~ 2.6936

