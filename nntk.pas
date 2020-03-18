/// Модуль для создания и обучений ИНС
unit nntk;
uses vector_math;

var 
  global_alpha: single;
  global_dropout_probability: single;
  global_initializing_weights_range: System.Tuple<real, real>;

type   
  functions_type = function(const input: Vector): Vector;

  // ********** Раздел функций активации и их производных **********
  Functions = class
    public
      /// Возвращает вектор, к каждому члену которого применена функция активации Линейный выпрямитель
      static function relu(const input: Vector): Vector;
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
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Линейный выпрямитель
      static function relu_derivative(const input: Vector): Vector;
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
    
      /// Возвращает вектор, к каждому члену которого применена Тождественная функция активации
      static function identity(const input: Vector): Vector;
      begin
        result := input
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная Тождественной функции активации
      static function identity_derivative(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := 1
      end;
        
      /// Возвращает вектор, к каждому члену которого применена функция активации Хевисайда
      static function binary_step(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          if input[index] > 0.5 then
            result[index] := 1
          else
            result[index] := 0;
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Хевисайда
      static function binary_step_derivative(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          if input[index] <> 0 then
            result[index] := 0
          else
            result[index] := System.Double.NaN;
      end;
      
      /// Возвращает вектор, к каждому члену которого применена функция активации Сигмоида
      static function sigmoid(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := 1/(1+exp(-input[index]))
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Сигмоида
      static function sigmoid_derivative(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := 1/(1+exp(-input[index]))*(1-1/(1+exp(-input[index])));
      end;
    
      /// Возвращает вектор, к каждому члену которого применена функции активации Гиперболический тангенс
      static function tanh(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := System.Math.Tanh(input[index]);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Гиперболический тангенс
      static function tanh_derivative(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := 1-System.Math.Tanh(input[index])**2;
      end;
      
      /// Возвращает вектор, к каждому члену которого применена функции активации Арктангенс
      static function arctan(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := System.Math.Atan(input[index]);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Арктангенс
      static function arctan_derivative(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := 1/(input[index]**2 + 1);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена функция активации Ареасинус
      static function arsinh(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := ln(input[index] + sqrt(input[index]**2 + 1));
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Ареасинус
      static function arsinh_derivative(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := 1/sqrt(input[index]**2 + 1);
      end;
            
      /// Возвращает вектор, к каждому члену которого применена функция активации Softsign
      static function softsign(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := input[index]/(1+abs(input[index]));
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Softsign
      static function softsign_derivative(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := 1/(1+abs(input[index]))**2;
      end;
            
      /// Возвращает вектор, к каждому члену которого применена функция активации Биполярный линейный выпрямитель
      static function brelu(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          if index mod 2 = 0 then
            result[index] := input[index]>0? input[index]: 0
          else
            result[index] := -(-input[index]>0? input[index]: 0);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Биполярный линейный выпрямитель
      static function brelu_derivative(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          if index mod 2 = 0 then
            result[index] := input[index]>0? 1: 0
          else
            result[index] := -input[index]>0? 1: 0;
      end;
      
      /// Возвращает вектор, к каждому члену которого применена функция активации Softplus
      static function softplus(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := ln(1+exp(input[index]));
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Softplus
      static function softplus_derivative(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := 1/(1+exp(-input[index]));
      end;
            
      /// Возвращает вектор, к каждому члену которого применена Выгнутая тождественная функция активации
      static function bent_identity(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := (sqr(input[index]**2+1)-1)/2 + input[index];
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная Выгнутой тождественной функции активации
      static function bent_identity_derivative(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := input[index]/(2*sqr(input[index]**2+1))+1;
      end;
      
      /// Возвращает вектор, к каждому члену которого применена функция активации Синусода
      static function sinusoid(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := sin(input[index]);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Синусода
      static function sinusoid_derivative(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := cos(input[index]);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена функция активации Кардинальный синус
      static function sinc(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := input[index]=0? 1: sin(input[index])/input[index];
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Кардинальный синус
      static function sinc_derivative(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := input[index]=0? 0: cos(input[index])/input[index] - sin(input[index])/input[index]**2;
      end;
            
      /// Возвращает вектор, к каждому члену которого применена Гауссова функция активации
      static function gaussian(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := exp(-(input[index]**2));
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная Гауссовой функции активации
      static function gaussian_derivative(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := -2*input[index]*exp(-(input[index]**2));
      end;
                  
      /// Возвращает вектор, к каждому члену которого применена Квадратная радиальная базисная функция активации
      static function sqrbf(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          if abs(input[index]) < 1 then
            result[index] := 1 - input[index]**2/2
          else if abs(input[index]) > 2 then
            result[index] := 0
          else
            result[index] := (2-abs(input[index]))**2/2;
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная Квадратной радиальной базисной функции активации
      static function sqrbf_derivative(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          if abs(input[index]) < 1 then
            result[index] := -input[index]
          else if abs(input[index]) > 2 then
            result[index] := 0
          else
            result[index] := input[index] - 2*sign(input[index]);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена функция активации Softmax
      static function softmax(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(input.size);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := exp(input[index]);
        var sum := result.sum();
        result /= sum;
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Softmax
      static function softmax_derivative(const input: Vector): Vector;
      begin
        result := softmax(input);
        {$omp parallel for}
        for var index := 0 to input.size-1 do
          result[index] := result[index]*(1-result[index]);
      end;
  end;

  Neuron = class
    private
      weights: Vector; 
      input: Vector;
      
      /// Иницилизирует number_of_weights случайных весов в диапазоне [0, 1)
      function initialize_weights(const number_of_weights: uint64): Vector;
      var
        tmp_result: array of single;
      begin
        tmp_result := new single[number_of_weights];
        {$omp parallel for}
        for var index := 0 to number_of_weights-1 do
          tmp_result[index] := random(global_initializing_weights_range[0],
                                      global_initializing_weights_range[1]);
        result := new Vector(tmp_result);
      end;
      
    public
      /// Инициализирует новый экземпляр класса Neuron с number_of_inputs входов
      constructor Create(const number_of_inputs: uint64);
      begin
        self.weights := initialize_weights(number_of_inputs);
      end;
      
      /// Возвращает ненормализированный выход нейрона
      function calculate(const input: Vector): single;
      begin
        self.input := input;
        result := self.weights.dot(self.input);
      end;

      /// Возвращает вектор ошибок для предшестующих нейронов 
      function backprop(const input: single): Vector;
      begin
        result := self.weights * input;
      end;
      
      /// Изменяет веса с учетом ошибки delta
      procedure adjust_weights(const delta: single);
      begin
        self.weights += self.input * delta * global_alpha;
      end;
      
      function ToString: string; override;
      begin
        result := 'Нейрон (Веса): ' + self.weights.ToString;
      end;
    end;

  Layer = class
    private
      layer: array of Neuron;
      
    public
      /// Инициализирует новый экземпляр класса Layer с number_of_neurons нейронов и number_of_weights входов для каждого
      constructor Create(const number_of_neurons, number_of_weights: uint64);
      begin
        if number_of_neurons < 1 then
          raise new System.ArgumentException('Размеры любого слоя ИНС должны быть больше 0');

        self.layer := new Neuron[number_of_neurons];
        {$omp parallel for}
        for var index := 0 to number_of_neurons-1 do
          self.layer[index] := new Neuron(number_of_weights);
      end;
      
      /// Возвращает ненормализированный вектор выходных значений слоя
      function calculate(const input: Vector): Vector;
      begin
        result := new Vector;
        result.set_size(self.layer.Length);
        {$omp parallel for}
        for var index := 0 to self.layer.length-1 do
          result[index] := self.layer[index].calculate(input); 
      end;
      
      /// Возвращает вектор ошибок для предшестующих слоев 
      function backprop(const input: Vector): Vector;
      begin
        result := self.layer[0].backprop(input[0]);
        {$omp parallel for reduction(+:result)}
        for var index := 1 to self.layer.Length-1 do
          result += self.layer[index].backprop(input[index]);
      end;
      
      /// Изменяет веса нейронов слоя с учетом ошибки delta
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

  Neural_Network = class
    private
      seed: integer;
    
      neural_network: array of Layer;
      topology: Vector;
      number_of_layers: uint64;
      activation_functions: array of function(const input: Vector): Vector;
      activation_functions_derivatives: array of nntk.functions_type;
      batch_size: uint64;

      /// Обучает нейронную сеть на входных данных input_data и выходных данных output_data number_of_epoch эпох 
      procedure __train(const input_data: List<Vector>; 
                        const output_data: List<Vector>;
                        const number_of_epoch: uint64);
      var
        deltas: array of Vector;
        layers: array of Vector;
        mask: array of Vector;
        error: real;
      begin
        deltas := new Vector[self.number_of_layers-1]; 
        {$omp parallel for}
        for var index := 0 to self.number_of_layers-2 do
          begin
          deltas[index] := new Vector;
          deltas[index].set_size(trunc(topology[self.number_of_layers-index-1]));
          end;
        layers := new Vector[self.number_of_layers];
        mask := new Vector[self.number_of_layers-1];

        for var epoch := 1 to number_of_epoch do
        begin 
          for var index := 0 to input_data.count-1 do
            begin
              layers[0] := input_data[index];
              for var i := 0 to self.number_of_layers-2 do
                layers[i+1] := self.activation_functions[i](self.neural_network[i].calculate(layers[i]));
              
              {$omp parallel for}
              for var i := 1 to self.number_of_layers-2 do
              begin
                mask[i-1] := get_dropout_mask(layers[i].size());
                layers[i] *= mask[i-1] * (1/(1-global_dropout_probability));
              end; 
                            
              if (epoch) mod 10 = 0 then
                error += ((output_data[index]-layers.last()) ** 2).sum()/output_data[index].size();
              
              deltas[0] += output_data[index]-layers.last();
              for var i := 1 to self.number_of_layers-2 do
              begin
                deltas[i] += self.neural_network[self.number_of_layers-i-1].backprop(deltas[i-1])
                           * self.activation_functions_derivatives[self.number_of_layers-i-1](layers[self.number_of_layers-i-1])
                           * mask[self.number_of_layers-i-2];
                end;
              if ((epoch-1)*input_data.Count+index+1) mod self.batch_size = 0 then
                {$omp parallel for}
                for var i := 0 to self.number_of_layers-2 do
                  self.neural_network[i].adjust_weights(deltas[self.number_of_layers-2-i]/self.batch_size);
                  deltas[self.number_of_layers-2-i] := new Vector;
                  deltas[self.number_of_layers-2-i].set_size(trunc(topology[i+1]));
            end;
          if (epoch) mod 10 = 0 then
            begin
            println('Error: ', error / input_data.count);
            error := 0.0;
            end;
          end;
      end;
      
    public
      /// Инициализирует объект класса Neural_Network с заданной топологией
      constructor Create(const neural_network_topology: Vector;
                         initializing_weights_range: System.Tuple<real, real> := (-1.0, 1.0);
                         activation_functions: array of nntk.functions_type := nil;
                         activation_functions_derivatives: array of nntk.functions_type := nil;
                         seed: integer := 0);
      begin
        Randomize(seed);
        self.topology := neural_network_topology;
                self.number_of_layers := neural_network_topology.size();
        if self.number_of_layers < 2 then
          raise new System.ArgumentException('Кол-во слоев ИНС должно быть больше 1');
        
        global_initializing_weights_range := initializing_weights_range;
        
        if activation_functions = nil then
          begin
          Setlength(self.activation_functions, self.number_of_layers-1);
          for var index := 0 to self.number_of_layers-2 do
            self.activation_functions[index] := nntk.Functions.sigmoid;
          end
        else if activation_functions.Length <> self.number_of_layers-1 then
          raise new System.ArgumentException('Кол-во функций активации и кол-во слоев ИНС данных должны совпадать')
        else
          begin
            Setlength(self.activation_functions, self.number_of_layers-1);
            for var index := 0 to self.number_of_layers-2 do
              self.activation_functions[index] := activation_functions[index];
          end;
          
        if activation_functions_derivatives = nil then
          begin
          Setlength(activation_functions_derivatives, self.number_of_layers-1);
          for var index := 0 to self.number_of_layers-2 do
            self.activation_functions_derivatives[index] := nntk.Functions.sigmoid_derivative;
          end
        else if activation_functions_derivatives.length <> self.number_of_layers-1 then
          raise new System.ArgumentException('Кол-во производных функций активации и кол-во слоев ИНС данных должны совпадать')
        else
          begin
            Setlength(self.activation_functions_derivatives, self.number_of_layers-1);
            for var index := 0 to self.number_of_layers-2 do
              self.activation_functions_derivatives[index] := activation_functions_derivatives[index];
          end;
          
        self.neural_network := new Layer[self.number_of_layers-1];
        {$omp parallel for}
        for var index := 1 to self.number_of_layers-1 do
          self.neural_network[index-1] := new Layer(trunc(neural_network_topology[index]),
                                                    trunc(neural_network_topology[index-1]));
        end;
      /// Обучает нейронную сеть на входных данных input_data и выходных данных output_data number_of_epoch эпох
      /// Необязательные параметры:
      /// Коэффициент обучаемости alpha (по умолчанию 0.01)
      /// Массив функций активации для разных слоев ИНС activation_functions (по умолчанию nntk.functions.relu) 
      /// Массив производных фукций активации для разных слоев ИНС activation_function_derivatives (по умолчанию nntk.functions.relu_derivative)
      /// Вероятность прореживания dropout_probability (по умолчанию 0.0 - Прореживание не проводится)
      /// Размер пакета разности весов batch_size (по умолчанию 1 - Обучение происходит на каждом отдельном примере)
      procedure train(const input_data: List<Vector>; 
                      const output_data: List<Vector>;
                      const number_of_epoch: uint64;
                      alpha: single := 0.01;
                      dropout_probability: single := 0.0; 
                      batch_size: uint64 := 1);
      begin
        if input_data.Count <> output_data.Count then
          raise new System.ArgumentException('Размеры обучающей выборки для входных и выходных данных должны совпадать');
        global_alpha := alpha;
        if (0 > dropout_probability) or (dropout_probability >= 1) then
          raise new System.ArgumentException('Вероятность прореживания узлов нейронной сети должна принадлежать [0, 1)');
        global_dropout_probability := dropout_probability;       
        if batch_size < 1 then
          raise new System.ArgumentException('Размер пакетов разности весов должен быть больше нуля');
        self.batch_size := batch_size;
        __train(input_data, output_data, number_of_epoch);  
      end;
      
      /// Обучает нейронную сеть на входных данных input_data и выходных данных output_data number_of_epoch эпох
      /// Необязательные параметры:
      /// Коэффициент обучаемости alpha (по умолчанию 0.01)
      /// Функция активации activation_function (по умолчанию nntk.functions.relu) 
      /// Производная фукции активации activation_function_derivative (по умолчанию nntk.functions.relu_derivative)
      /// Вероятность прореживания dropout_probability (по умолчанию 0.0 - Прореживание не проводится)
      /// Размер пакета разности весов batch_size (по умолчанию 1 - Обучение происходит на каждом отдельном примере)
      procedure train(const input_data: array of Vector; 
                      const output_data: array of Vector;
                      const number_of_epoch: uint64;
                      alpha: single := 0.01;
                      dropout_probability: single := 0.0;
                      batch_size: uint64 := 1);
                      
      begin
        if input_data.Length <> output_data.Length then
          raise new System.ArgumentException('Размеры обучающей выборки для входных и выходных данных должны совпадать');
        global_alpha := alpha;
        if (0 > dropout_probability) or (dropout_probability >= 1) then
          raise new System.ArgumentException('Вероятность прореживания узлов нейронной сети должна принадлежать [0, 1)');
        global_dropout_probability := dropout_probability;       
        if batch_size < 1 then
          raise new System.ArgumentException('Размер пакетов разности весов должен быть больше нуля');
        self.batch_size := batch_size;        
        __train(new List<Vector>(input_data),
                new List<Vector>(output_data), 
                number_of_epoch);
      end;

      /// Возвращает результат работы нейронной сети для входных данных input_data
      function run(const input_data: Vector): Vector;
      begin
      if input_data.size() <> self.topology[0] then
          raise new System.ArgumentException('Размеры первого слоя ИНС и входных данных должны совпадать');  
      var layers := new Vector[self.number_of_layers];
      layers[0] := input_data;
      for var i := 0 to self.number_of_layers-2 do
        layers[i+1] := self.activation_functions[i](self.neural_network[i].calculate(layers[i]));
      result := layers[self.number_of_layers-1];
      end;
      
      /// Возвращает модель нейронной сети в виде функции
      function get_model(): function (input_data: Vector): Vector;
      begin
        result := self.run;
      end;
      
      /// Возвращает модель вектор прореживания
      function get_dropout_mask(const size: uint64): Vector;
      begin
        result := new Vector;
        result.set_size(size);
        {$omp parallel for}
        for var index := 0 to size-1 do
          result[index] := random() < global_dropout_probability? 0: 1;
      end;
  end;
end.