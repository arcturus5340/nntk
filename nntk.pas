unit nntk;
uses vector_math;

type 
  Neuron = class
    private
      weights: Vector; 
      input: Vector;
      alpha: real = 0.01;
      
      function initialize_weights(number_of_weights: integer): Vector;
      begin
        result := new Vector;
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
      
      function calculate(input: Vector): real;
      begin
        self.input := input;
        result := self.weights.dot(self.input);
      end;

      function backprop(input: real): Vector;
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
      layer: array of Neuron;
      
    public
      constructor Create(number_of_neurons: integer; 
                         number_of_weights: integer);
      begin
        self.layer := new Neuron[number_of_neurons];
        for var index := 0 to number_of_neurons-1 do
          self.layer[index] := new Neuron(number_of_weights);
      end;
      
      function calculate(input: Vector): Vector;
      begin
        result := new Vector;
        for var index := 0 to self.layer.Count-1 do
        begin
          result.push_back(self.layer[index].calculate(input)) 
        end;
      end;
      
      function backprop(input: Vector): Vector;
      begin
        result := self.layer[0].backprop(input[0]);
        for var index := 1 to self.layer.Count-1 do
        begin
          result := result + self.layer[index].backprop(input[index]);
        end;
      end;
      
      procedure adjust_weights(delta: Vector);
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
      neural_network: array of Layer;
      number_of_layers: integer;
      
    public
      constructor Create(neural_network_topology: Vector);
      begin
        self.number_of_layers := neural_network_topology.size();
        self.neural_network := new Layer[number_of_layers-1];
        for var index := 1 to number_of_layers-1 do
          self.neural_network[index-1] := new Layer(trunc(neural_network_topology[index]), 
                                                    trunc(neural_network_topology[index-1]));
      end;
      
      function run(input_data: Vector): Vector;
      begin
      var layers := new Vector[self.number_of_layers];
      layers[0] :=input_data;
      for var i := 0 to self.number_of_layers-2 do
        layers[i+1] := activation_function(self.neural_network[i].calculate(layers[i]));
      result := layers[self.number_of_layers-1];
      end;
   
      function get_model(): function (input_data: Vector): Vector;
      begin
        result := self.run;
      end;
      
      // TrainingSetSize Exception
      procedure learn(input_data: List<Vector>; 
                      output_data: List<Vector>;
                      number_of_epoch: integer);
      var
        deltas: array of Vector;
        layer: array of Vector;
        error: real;
      begin
        deltas := new Vector[self.number_of_layers-1];
        println(neural_network);
        for var epoch := 1 to number_of_epoch do
          begin
          for var index := 0 to input_data.Count-1 do
            begin
              var layers := new Vector[self.number_of_layers]; 
              layers[0] := input_data[index];
              for var i := 0 to self.number_of_layers-2 do
                layers[i+1] := activation_function(self.neural_network[i].calculate(layers[i]));
              
              if epoch mod 10 = 0 then
                error += ((output_data[index]-layers.last()) ** 2).sum();
              
              deltas[0] := output_data[index]-layers.last();
              for var i := 1 to self.number_of_layers-2 do
                deltas[i] := self.neural_network[self.number_of_layers-i-1].backprop(deltas[i-1])
                                               * activation_function_derivative(layers[self.number_of_layers-i-1]);
//              println('Deltas: ', deltas);

              for var i := 0 to self.number_of_layers-2 do
                self.neural_network[i].adjust_weights(deltas[self.number_of_layers-2-i]);
            end;
          if epoch mod 10 = 0 then
            begin
            println('Error: ', error / input_data.Count);
            error := 0.0;
            end;
          end;
      end;
            
      function activation_function(input: Vector): Vector;
      begin
        result := new Vector;
        for var index := 0 to input.size-1 do
          if input[index] > 0 then
            result.push_back(input[index])
          else
            result.push_back(0);
      end; 

      function activation_function_derivative(input: Vector): Vector;
      begin
        result := new Vector;
        for var index := 0 to input.size-1 do
          if input[index] > 0 then
            result.push_back(1)
          else
            result.push_back(0);
      end;
  end;
end.