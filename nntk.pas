unit nntk;
uses vector_math;

type 
  Neuron = class
    private
      weights: Vector<real>; 
      input: Vector<real>;
      alpha: real = 0.01;
      
      function initialize_weights(number_of_weights: integer): Vector<real>;
      begin
        result := new Vector<real>;
        for var index := 0 to number_of_weights-1 do
        begin
          // Random weight [0, 1)
          result.push_back(random*2-1);
        end;
//        println(result);
      end;

      procedure adjust_weights(delta:real);
      begin
        var new_delta := self.input * delta;
        new_delta := new_delta * alpha;
        self.weights := self.weights + new_delta;
      end;
      
    public
      constructor Create(number_of_inputs: integer);
      begin
        self.weights := initialize_weights(number_of_inputs);
      end;
      
      function calculate(input: Vector<real>): real;
      begin
        self.input := input;
        result := self.weights.dot(self.input);
      end;

      function backprop(input: real): Vector<real>;
      begin
        result := self.weights * input;
      end;
      
      function ToString: string; override;
      begin
        result := 'Нейрон (Веса): ' + self.weights.ToString;
      end;
  end;

type 
  Layer = class
    private
      layer: List<Neuron>;
      
    public
      constructor Create(number_of_neurons: integer; 
                         number_of_weights: integer);
      begin
        self.layer := new list<Neuron>;
        for var index := 0 to number_of_neurons-1 do
          self.layer.add(new Neuron(number_of_weights));
      end;
      
      function calculate(input: Vector<real>): Vector<real>;
      begin
        result := new Vector<real>;
        for var index := 0 to self.layer.Count-1 do
        begin
          result.push_back(self.layer[index].calculate(input)) 
        end;
      end;
      
      function backprop(input: Vector<real>): Vector<real>;
      begin
        result := self.layer[0].backprop(input[0]);
        for var index := 1 to self.layer.Count-1 do
        begin
          result := result + self.layer[index].backprop(input[index]);
        end;
      end;
      
      procedure adjust_weights(delta: Vector<real>);
      begin
        for var index := 0 to self.layer.Count-1 do
        begin  
//          println('Weights <: ', self.layer[index].weights);
          self.layer[index].adjust_weights(delta[index]);
//          println('Weights >: ', self.layer[index].weights);
          end;
      end;
      
      function ToString: string; override;
      begin
        result := 'Слой: ';
        for var index := 0 to layer.Count-2 do
          result += layer[index].ToString + ' | ';
        result += layer[layer.Count-1].ToString;
      end;
  end;

type
  Neural_Network = class
    private
      neural_network: List<Layer>;
      number_of_layers: integer;
      
    public
      constructor Create(neural_network_topology: Vector<integer>);
      begin
        self.neural_network := new List<Layer>;
        self.number_of_layers := neural_network_topology.size();
        for var index := 1 to number_of_layers-1 do
          self.neural_network.add(new Layer(neural_network_topology[index], 
                                            neural_network_topology[index-1]));
      end;
      
      function run(input_data: Vector<real>): Vector<real>;
      begin
      var layers := new Vector<Vector<real>>(input_data);
      for var i := 0 to self.number_of_layers-2 do
        layers.push_back(activation_function(self.neural_network[i].calculate(layers.back())));
      result := layers.back();
      end;
   
      function get_model(): function (input_data: Vector<real>): Vector<real>;
      begin
        result := self.run;
      end;
      
      // TrainingSetSize Exception
      procedure learn(input_data: Vector<Vector<real>>; 
                      output_data: Vector<Vector<real>>;
                      number_of_epoch: integer);
      begin
        for var epoch := 0 to number_of_epoch-1 do
          for var index := 0 to input_data.size-1 do
            begin
              var layers := new Vector<Vector<real>>(input_data[index]);
              for var i := 0 to self.number_of_layers-2 do
                layers.push_back(activation_function(self.neural_network[i].calculate(layers.back())));
//              println('Layers: ', layers);
              println('Error: ', (output_data[index]-layers.back()) ** 2, '[', layers.back(), ']');

              var deltas := new Vector<Vector<real>>(output_data[index]-layers.back());
              for var i := 1 to self.number_of_layers-2 do
                begin
                deltas.push_back(self.neural_network[self.number_of_layers-i-1].backprop(deltas.back())
                               * activation_function_derivative(layers[self.number_of_layers-i-1]));
                end;  
//              println('Deltas: ', deltas);

              for var i := 0 to self.number_of_layers-2 do
                self.neural_network[i].adjust_weights(deltas[self.number_of_layers-2-i]);
            end;
      end;
            
      function activation_function(input: Vector<real>): Vector<real>;
      begin
        result := new Vector<real>;
        for var index := 0 to input.size-1 do
          if input[index] > 0 then
            result.push_back(input[index])
          else
            result.push_back(0);
      end; 

      function activation_function_derivative(input: Vector<real>): Vector<real>;
      begin
        result := new Vector<real>;
        for var index := 0 to input.size-1 do
          if input[index] > 0 then
            result.push_back(1)
          else
            result.push_back(0);
      end;
  end;
end.