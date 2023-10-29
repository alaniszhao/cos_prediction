#include "Arduino.h"
#include "MicroFlow_Q.h"

MicroMLP::MicroMLP(int la, int* top, int* w, int* b, int* a){
  layers = la;
  topology = top;
  weights = w;
  biases = b;
  activations = a;
}

MicroMLP::MicroMLP(int la, int* top, int* w, int* b, int a){
  layers = la;
  topology = top;
  weights = w;
  biases = b;
  sameActiv = a;
  allSameActiv = true;
}

void activate(int l, int* z, int activation) {
  for (int i = 0; i < l; i++) {
    if (activation == SIGMOID) {
      z[i] = 1 / (1 + exp(-z[i]));
    } else if (activation == TANH) {
      z[i] = tanh(z[i]);
    } else if (activation == EXPONENTIAL){
      z[i] = exp(z[i]);
    } else if (activation == SWISH){
      z[i] = z[i] / (1 + exp(-z[i]));
    } else if (activation == RELU){
      z[i] = fmax(0, z[i]);
    }
  }
}

void MicroMLP::feedforward(int* input, int* out){
  int maxLayer = 0;
  for (int i=0;i<layers;i++){
    if (topology[i] > maxLayer){
      maxLayer = topology[i];
    }
  }

  int x[maxLayer];
  for (int i=0;i<topology[0];i++){
    x[i] = input[i];
  }
  int weightAdder = 0;
  int biasAdder = 0;
  for (int l=0;l<layers-1;l++){

    //Matrix----
    int cpy[topology[l]];
    for (int i=0;i<topology[l];i++){
      cpy[i] = x[i];
    }
    int columnB = 0;
    for (int i = 0; i < topology[l+1]; i++) {
      int bi = columnB;
      int sum = 0;
      
      for (int j = 0; j < topology[l]; j++) {
        sum += cpy[j] * weights[bi+weightAdder];
        bi += topology[l+1];
      }
      x[i] = sum;
      columnB ++;
    }
    
    for (int i=0;i<topology[l+1];i++){
      x[i] += biases[i+biasAdder];
    }
    if (l != layers-2){
      if (!allSameActiv)
        activate(topology[l+1], x, activations[l]);
      else
        activate(topology[l+1], x, sameActiv);
    }
    weightAdder += topology[l]*topology[l+1];
    biasAdder += topology[l+1];
  }
  for (int i=0;i<topology[layers-1];i++){
    out[i] = x[i];
  }
}
