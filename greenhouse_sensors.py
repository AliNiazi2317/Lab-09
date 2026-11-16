
import numpy as np

np.random.seed(7)

readings = np.random.randint(20, 41, size=(6, 3))

print(readings)

sensor_mean = readings.mean(axis=0)
print("Mean per sensor:", sensor_mean)

sensor_std = readings.std(axis=0)
normalized = (readings - sensor_mean) / sensor_std

print("Normalized column means:", normalized.mean(axis=0).round(1))

peak_hours = readings.argmax(axis=0)
print("Hour of max per sensor:", peak_hours)

mean_row = sensor_mean.reshape(1, 3)
readings_with_mean = np.vstack([readings, mean_row])

print(readings_with_mean.shape)

np.random.seed(7)
readings_again = np.random.randint(20, 41, size=(6, 3))

print("Reproducible:", np.array_equal(readings, readings_again))
print("Overall hottest reading:", readings.max())