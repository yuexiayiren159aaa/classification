def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x



# for i in range(255):
#     print(preprocess_input(i))

import os

print(os.listdir("./"))

