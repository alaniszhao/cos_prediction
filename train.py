import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import math
"""
x = np.arange(0, math.pi*2, .1)
y = (np.sin(x)+1)/2 

model = Sequential([
    Dense(10, input_shape=(1,)),
    Activation('sigmoid'),
    Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['mean_squared_error'])
model.fit(x, y, epochs=100000, batch_size=8, verbose=0)

preds = model.predict(x)

plt.plot(x, y, 'b', x, preds, 'r--')
plt.ylabel('Y / Predicted Value')
plt.xlabel('X Value')
plt.show()

x = np.arange(0, 100, .1)
y = (np.sin(x)+1)/2 

model_copy = model
model_copy.fit(x, y, epochs=10000, batch_size=8, verbose=0)
model_copy_preds = model_copy.predict(x)

plt.plot(x, y, 'b', x, model_copy_preds, 'r--')
plt.ylabel('Y / Predicted Value')
plt.xlabel('X Value')
plt.show()
print(model.layers[0].get_weights()[0])
print(model.layers[0].get_weights()[1])
print(model.layers[1].get_weights()[0])
print(model.layers[1].get_weights()[1])
"""

x = np.arange(-3*np.pi, 3*np.pi, 0.01).reshape(-1,1)
y = np.cos(x)

model = Sequential()
#model.add(Dense(1))
#model.add(Dense(16, activation='tanh'))
#model.add(Dense(16, activation='tanh'))
#model.add(Dense(1))
#model.compile(optimizer='adam', loss='mse')
model = Sequential()
model.add(Dense(16, activation='tanh',input_shape=(1,)))
model.add(Dense(16, activation='tanh'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['mean_squared_error'])

for i in range(10):
    model.fit(x, y, epochs=80, batch_size=10, verbose=1)
    predictions = model.predict(x)

model.compile(loss='mse', optimizer='adam')

preds = model.predict(x)
plt.plot(x, y, 'b', x, preds, 'r--')
plt.ylabel('Y / Predicted Value')
plt.xlabel('X Value')
#plt.show()
"""
print(model.layers[0].get_weights()[1])
print(model.layers[1].get_weights()[1])
print(model.layers[2].get_weights()[1])
print(model.layers[3].get_weights()[1])
#print(model.layers[0].get_weights()[1])
#print(model.layers[1].get_weights()[1])
"""
def weights_to_cpp(model, filename="weights_and_biases.txt"):
    model.summary()
    weights = []
    biases = []
    for l in range(len(model.layers)):
        W, B = model.layers[l].get_weights()
        weights.append(W.flatten())
        biases.append(B.flatten())
    print(weights)
    z = []
    b = []
    for i in weights:
        for l in i:
            z.append(l)
    for i in biases:
        for l in i:
            b.append(l)
    with open(filename, "w") as f:
      f.write("weights: {")
      for i in range(len(z)):
        if (i < len(z)-1):
          f.write(str(z[i])+", ")
        else:
          f.write(str(z[i]))
      f.write("}\n\n")

      f.write("biases: {")
      for i in range(len(b)):
        if (i < len(b)-1):
          f.write(str(b[i])+", ")
        else:
          f.write(str(b[i]))
      f.write("}\n\n")
    
      arch = []
    
      arch.append(model.layers[0].input_shape[1])
      for i in range(1, len(model.layers)):
          arch.append(model.layers[i].input_shape[1])
      arch.append(model.layers[len(model.layers)-1].output_shape[1])
      f.write("Architecture: {")
      for i in range(len(arch)):
          if (i < len(arch)-1):
              f.write(str(arch[i])+", ")
          else:
              f.write(str(arch[i]))
      f.write("}")
      print("Architecture (alpha):", arch)
      print("Layers: ", len(arch))
    print("Weights: ", z)
    print("Biases: ", b)

weights_to_cpp(model, filename="w.txt")