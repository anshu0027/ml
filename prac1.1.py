#blank array
import numpy as np
print('IU2141230160')
blank_array = np.zeros((3, 3))
print("Blank Array:")
print(blank_array)

import numpy as np
print("IU2141230160")
predefined_array = np.array([1, 3, 5, 7, 9])
print("Predefined Array:")
print(predefined_array)

import numpy as np
print("IU2141230160")
pattern_array = np.ones((3, 3))
print("Pattern-specific Array:")
print(pattern_array)

import numpy as np
print("IU2141230160")
predefined_array = np.array([1, 3, 6, 7, 9])
sliced_array = predefined_array[1:4]
print("Sliced Array:")
print(sliced_array)
predefined_array[2] = 10
print("Updated Predefined Array:")
print(predefined_array)

import numpy as np
print("IU2141230160")
predefined_array = np.array([1, 3, 5, 6, 8])
reshaped_array = np.reshape(predefined_array, (5, 1))
print("Reshaped Array:")
print(reshaped_array)
flattened_array = reshaped_array.flatten()
print("Flattened Array:")
print(flattened_array)

import numpy as np
print("IU2141230160")
predefined_array = np.array([1, 3, 4, 6, 7])
print("Looping Over Predefined Array:")
for element in predefined_array:
  print(element)

import numpy as np
data = np.loadtxt('data.txt', delimiter=',')
print("Data from file:")
print(data)
