#include "MicroFlow_Q.h"
void setup(){
  Serial.begin(9600);
  while(!Serial){
    ;
  }
  int topology[] = {1,16,16,1};
  int layers = 4;
  double inputs[] = {0};
  double output[1] = {};
  int weights[]={60,18,18,-53,34,-68,23,21,56,-67,-59,-42,10,22,21,13,-56,33,13,18,-24,11,31,39,33,5,-29,33,3,11,36,33,83,0,-53,-21,-9,43,-22,38,-11,-5,-20,66,40,47,15,28,86,38,-12,4,-20,31,32,-23,31,-6,27,60,-34,15,20,25,24,1,18,-45,-48,-41,-35,6,30,26,39,23,34,33,44,22,105,-6,-122,12,-33,-58,34,-149,-46,-38,-14,-25,14,4,-53,41,46,39,-29,-35,12,16,-11,19,-5,8,41,36,23,1,34,-28,81,8,-1,33,17,31,-32,45,-12,26,-7,72,-11,44,54,4,2,24,-51,-40,-29,25,9,-3,2,24,-16,-41,-15,-2,-41,-11,-18,23,-22,24,58,42,-32,37,53,53,17,46,-18,43,-20,37,89,61,-16,15,25,49,-3,-3,52,12,-57,-77,24,53,-21,-28,34,-8,-12,-27,16,41,-37,33,55,-31,70,17,-2,-23,-13,10,21,-35,-22,40,22,-36,-15,-27,-20,-23,39,14,-18,17,-19,-21,50,9,-19,29,7,2,-1,-40,7,-10,2,24,23,10,-13,26,51,9,-82,-27,-19,-20,40,-54,-62,3,-47,11,10,13,-44,37,80,38,-43,-3,37,28,28,40,16,20,24,55,11,-41,1,-7,34,-14,-81,41,49,18,-12,-80,33,-29,-13,55,-37,28,28,15,-137,-49,161,5,-47,-76,26,-117,-86,-56,-64,193,-6,-47,-29,-7};
  int biases[] = {-9,-102,-98,-39,246,-56,-150,3,-105,233,-99,10,-4,110,-123,41,-39,2,-15,-2,40,34,22,16,54,20,-49,-37,7,10,18,2,-109};
  MicroMLP mlp(layers, topology, weights, biases, TANH);
  for (int i=0;i<180;i++){
    inputs[0] = i * (3.14/180);
    //Feedforward pass through the network
    mlp.feedforward(inputs, output);
    Serial.print("Inputs: ");Serial.println(inputs[0]);
    Serial.print("Neural Network Output: ");Serial.println(output[0]);
    Serial.print("Actual:");Serial.println(cos(inputs[0]));
  }
}


void loop(){
  
}
