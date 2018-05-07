import numpy as np

relu = np.vectorize(lambda x: 0 if x<0 else x)

class WeightLayer:

    def init(self, num_inputs, num_outputs, activ = relu):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.activ = activ
        self.weight_matrix = np.random.rand((num_inputs + 1, num_outputs))
    
    def compute(self, input_vec):
        with_bias = np.concatenate(input_vec,np.array([1]))
        return self.activ(self.weight_matrix*input_vec)

    def set_weights(self,new):
        self.weights = new

class NerualNet():
    
    def init(self):
        self.weight_layers = []
        self.num_layers = 0
    
    def add_layer(layer):
        self.weight_layers.append(layer)
        self.num_layers += 1

    def compute(self, input_vec):
        latest = input_vec
        for wl in self.weight_layers:
            latest = wl.compute(latest)
        return latest


        
        
        
