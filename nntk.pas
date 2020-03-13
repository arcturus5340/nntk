﻿unit nntk;
uses vector_math;

type 
  Neuron = class
    private
      weights: Vector; 
      input: Vector;
      alpha: single = 0.01;
      
      function initialize_weights(const number_of_weights: integer): Vector;
      var
        tmp_result: array of single;
      begin
        tmp_result := new single[number_of_weights];
        {$omp parallel for}
        for var index := 0 to number_of_weights-1 do
          // Random weight [0, 1)
          tmp_result[index] := random*2-1;
        result := new Vector(tmp_result);
      end;

      procedure adjust_weights(const delta: single);
      begin
        self.weights := self.weights + self.input * delta * alpha;
      end;
      
    public
      constructor Create(const number_of_inputs: integer);
      begin
        self.weights := initialize_weights(number_of_inputs);
      end;
      
      function calculate(const input: Vector): single;
      begin
        self.input := input;
        result := self.weights.dot(self.input);
      end;

      function backprop(const input: single): Vector;
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
      constructor Create(const number_of_neurons, number_of_weights: integer);
      begin
        self.layer := new Neuron[number_of_neurons];
        {$omp parallel for}
        for var index := 0 to number_of_neurons-1 do
          self.layer[index] := new Neuron(number_of_weights);
      end;
      
      function calculate(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(self.layer.Length);
        {$omp parallel for}
        for var index := 0 to self.layer.length-1 do
          result[index] := self.layer[index].calculate(input); 
      end;
      
      function backprop(const input: Vector): Vector;
      begin
        result := self.layer[0].backprop(input[0]);
        {$omp parallel for reduction(+:result)}
        for var index := 1 to self.layer.Length-1 do
          result := result + self.layer[index].backprop(input[index]);
      end;
      
      procedure adjust_weights(const delta: Vector);
      begin
        {$omp parallel for}
        for var index := 0 to self.layer.Length-1 do
          self.layer[index].adjust_weights(delta[index]);
      end;
      
      function ToString: string; override;
      begin
        result := 'Слой: ';
        for var index := 0 to layer.Length-2 do
          result += layer[index].ToString + ' | ';
        result += layer[layer.Length-1].ToString;
      end;
  end;

type
  Neural_Network = class
    private
      neural_network: array of Layer;
      number_of_layers: integer;
      
      procedure __train(const input_data: List<Vector>; 
                        const output_data: List<Vector>;
                        const number_of_epoch: integer);
      var
        deltas: array of Vector;
        layers: array of Vector;
        mask: array of Vector;
        error: single;
      begin
        deltas := new Vector[self.number_of_layers-1];
        layers := new Vector[self.number_of_layers]; 
        mask := new Vector[self.number_of_layers-1];

        for var epoch := 1 to number_of_epoch do
        begin 
          for var index := 0 to input_data.count-1 do
            begin
              layers[0] := input_data[index];
              {$omp parallel for}
              for var i := 0 to self.number_of_layers-2 do
                begin
                layers[i+1] := activation_function(self.neural_network[i].calculate(layers[i]));
                mask[i] := dropout_mask(layers[i+1].size());
                layers[i+1] := layers[i+1] * mask[i] * 2;
                end;
              
              if epoch mod 10 = 0 then
                error += ((output_data[index]-layers.last()) ** 2).sum();
                      
              deltas[0] := output_data[index]-layers.last();
              {$omp parallel for}
              for var i := 1 to self.number_of_layers-2 do
                deltas[i] := self.neural_network[self.number_of_layers-i-1].backprop(deltas[i-1])
                           * activation_function_derivative(layers[self.number_of_layers-i-1])
                           * mask[self.number_of_layers-i-2];
//              println('Deltas: ', deltas);
              {$omp parallel for}
              for var i := 0 to self.number_of_layers-2 do
                self.neural_network[i].adjust_weights(deltas[self.number_of_layers-2-i]);
            end;
          if epoch mod 10 = 0 then
            begin
            println('Error: ', error / input_data.count);
            error := 0.0;
            end;
          end;
      end;
      
    public
      constructor Create(const neural_network_topology: Vector);
      begin
        self.number_of_layers := neural_network_topology.size();
        self.neural_network := new Layer[number_of_layers-1];
        {$omp parallel for}
        for var index := 1 to number_of_layers-1 do
          self.neural_network[index-1] := new Layer(trunc(neural_network_topology[index]), 
                                                    trunc(neural_network_topology[index-1]));
      end;
      
      procedure train(const input_data: List<Vector>; 
                      const output_data: List<Vector>;
                      const number_of_epoch: integer);
      begin
        __train(input_data, output_data, number_of_epoch);  
      end;
      procedure train(const input_data: array of Vector; 
                      const output_data: array of Vector;
                      const number_of_epoch: integer);
      begin
        __train(new List<Vector>(input_data), 
                new List<Vector>(output_data), 
                number_of_epoch);
      end;

      function run(const input_data: Vector): Vector;
      begin
      var layers := new Vector[self.number_of_layers];
      layers[0] :=input_data;
      {$omp parallel for}
      for var i := 0 to self.number_of_layers-2 do
        layers[i+1] := activation_function(self.neural_network[i].calculate(layers[i]));
      result := layers[self.number_of_layers-1];
      end;
   
      function get_model(): function (input_data: Vector): Vector;
      begin
        result := self.run;
      end;
      
      function activation_function(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          if input[index] > 0 then
            result[index] := input[index]
          else
            result[index] := 0;
      end; 

      function activation_function_derivative(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          if input[index] > 0 then
            result[index] := 1
          else
            result[index] := 0;
      end;
      
      function dropout_mask(const size: integer): Vector;
      begin
        result := new Vector;
        result.set_size(size);
        while result.count(1) <> (size div 2) do
          {$omp parallel for}
          for var index := 0 to size-1 do
            result[index] := random(2);
      end;
  end;
end.