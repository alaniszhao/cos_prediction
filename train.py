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
import tensorflow

x = np.arange(-5*np.pi, 5*np.pi, 0.001).reshape(-1,1)
y = np.cos(x)
#,kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0, stddev=100.0, seed=None)
model = Sequential()
model = Sequential()
model.add(Dense(16, activation='tanh',input_shape=(1,),kernel_initializer=tensorflow.keras.initializers.RandomUniform(minval=-5., maxval=5.)))
model.add(Dense(16, activation='tanh',kernel_initializer=tensorflow.keras.initializers.RandomUniform(minval=-5., maxval=5.)))
model.add(Dense(16, activation='tanh',kernel_initializer=tensorflow.keras.initializers.RandomUniform(minval=-5., maxval=5.)))
model.add(Dense(1,kernel_initializer=tensorflow.keras.initializers.RandomUniform(minval=-5., maxval=5.)))
model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['mean_squared_error'])

for i in range(60):
    model.fit(x, y, epochs=100, batch_size=100, verbose=1)
    predictions = model.predict(x)

preds = model.predict(x)
plt.plot(x, y, 'b', x, preds, 'r--')
plt.ylabel('Y / Predicted Value')
plt.xlabel('X Value')
plt.show()

def weights_to_cpp(model, filename="weights_and_biases.txt"):
    #model.summary()
    weights = []
    biases = []
    for l in range(len(model.layers)):
        if l==0:
           continue
        W=[]
        B=[]
        for c in model.layers[l].get_weights():
          if (len(W)==0):
            W=c
          elif (len(B)==0):
            B=c
          else:
            break
        #W, B = model.layers[l].get_weights()
        weights.append(W.flatten())
        biases.append(B.flatten())
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