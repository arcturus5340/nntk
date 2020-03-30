/// Модуль для создания и обучений ИНС
unit nntk;
uses neo;

var 
  global_alpha: real;
  global_dropout_probability: real;
  global_initializing_weights_range: System.Tuple<real, real>;

type   
  functions_type = function(const input: neo.neo_array): neo.neo_array;

  // ********** Раздел функций активации и их производных **********
  functions = class
    private   
      static elu_alpha: real;
      static elu_derivative_alpha: real;       
      static isru_alpha: real;
      static isru_derivative_alpha: real;      
      static isrlu_alpha: real;
      static isrlu_derivative_alpha: real;
      static plu_alpha: real;
      static plu_c: real;
      static plu_derivative_alpha: real;
      static plu_derivative_c: real;
      static prelu_alpha: real;
      static prelu_derivative_alpha: real;
      static softexponential_alpha: real;
      static softexponential_derivative_alpha: real;
      static srelu_al: real;
      static srelu_ar: real;
      static srelu_tl: real;
      static srelu_tr: real;
      static srelu_derivative_al: real;
      static srelu_derivative_ar: real;
      static srelu_derivative_tl: real;
      static srelu_derivative_tr: real;      
      
      /// Возвращает вектор, к каждому члену которого применена функции активации Арктангенс
      static function __arctan(const input: real): real;
      begin
        result := System.Math.Atan(input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Арктангенс
      static function __arctan_derivative(const input: real): real;
      begin
        result := 1/(input**2 + 1);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена функция активации Ареасинус
      static function __arsinh(const input: real): real;
      begin
        result := ln(input + sqrt(input**2 + 1));
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Ареасинус
      static function __arsinh_derivative(const input: real): real;
      begin
        result := 1/sqrt(input**2 + 1);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена Выгнутая тождественная функция активации
      static function __bent_identity(const input: real): real;
      begin
        result := (sqr(input**2+1)-1)/2 + input;
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная Выгнутой тождественной функции активации
      static function __bent_identity_derivative(const input: real): real;
      begin
        result := input/(2*sqr(input**2+1))+1;
      end;
 
      /// Возвращает вектор, к каждому члену которого применена функция активации Хевисайда
      static function __binary_step(const input: real): real;
      begin
        if input < 0 then
          result := 0
        else
          result := 1;
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Хевисайда
      static function __binary_step_derivative(const input: real): real;
      begin
        if input <> 0 then
          result := 0
        else
          result := System.Double.NaN;
      end;
      
      static function __elu(const input: real): real;
      begin
        if input > 0 then
          result := input
        else
          result := elu_alpha*(exp(input)-1)
      end;
 
      static function __elu_derivative(const input: real): real;
      begin
        if input > 0 then
          result := 1
        else
          result :=  elu_derivative_alpha*(exp(input)-1)+elu_derivative_alpha;
      end;
            
      /// Возвращает вектор, к каждому члену которого применена Гауссова функция активации
      static function __gaussian(const input: real): real;
      begin
        result := exp(-(input**2));
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная Гауссовой функции активации
      static function __gaussian_derivative(const input: real): real;
      begin
        result := -2*input*exp(-(input**2));
      end;
      
      /// Возвращает вектор, к каждому члену которого применена Тождественная функция активации
      static function __identity(const input: real): real;
      begin
        result := input
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная Тождественной функции активации
      static function __identity_derivative(const input: real): real;
      begin
        result := 1
      end;
                  
      static function __isru(const input: real): real;
      begin
        result := input/sqrt(1 + isru_alpha*input**2);
      end;
 
      static function __isru_derivative(const input: real): real;
      begin
        result := (1/sqrt(1 + isru_derivative_alpha*input**2))**3;
      end;
      
      static function __isrlu(const input: real): real;
      begin
        if input < 0 then
          result := input/sqrt(1 + isrlu_alpha*input**2)
        else
          result := input
      end;
 
      static function __isrlu_derivative(const input: real): real;
      begin
        if input < 0 then
          result := (1/sqrt(1 + isrlu_derivative_alpha*input**2))**3
        else
          result := 1;
      end;

      static function __plu(const input: real): real;
      begin
        result := max(plu_alpha*(input+plu_c)-plu_c, min(plu_alpha*(input-plu_c)+plu_c, input));
      end;
 
      static function __plu_derivative(const input: real): real;
      begin
        if abs(input) > plu_derivative_c then
          result := plu_derivative_alpha
        else
          result := 1;
      end;               

      static function __prelu(const input: real): real;
      begin
        if input < 0 then
          result := prelu_alpha*input
        else
          result := input
      end;
 
      static function __prelu_derivative(const input: real): real;
      begin
        if input < 0 then
          result := prelu_derivative_alpha
        else
          result := 1;
      end;               
      
      /// Возвращает вектор, к каждому члену которого применена функция активации Линейный выпрямитель
      static function __relu(const input: real): real;
      begin
        if input > 0 then
          result := input
        else 
          result := 0;
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Линейный выпрямитель
      static function __relu_derivative(const input: real): real;
      begin
        if input > 0 then
          result := 1
        else
          result := 0;
      end;
      
      /// Возвращает вектор, к каждому члену которого применена функция активации Сигмоида
      static function __sigmoid(const input: real): real;
      begin
        result := 1/(1+exp(-input))
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Сигмоида
      static function __sigmoid_derivative(const input: real): real;
      begin
        result := 1/(1+exp(-input))*(1-1/(1+exp(-input)));
      end;
                  
      /// Возвращает вектор, к каждому члену которого применена функция активации Кардинальный синус
      static function __sinc(const input: real): real;
      begin
        if input = 0 then
          result := 1
        else
          result := sin(input)/input;
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Кардинальный синус
      static function __sinc_derivative(const input: real): real;
      begin
        if input = 0 then
          result := 0
        else
          result := cos(input)/input - sin(input)/input**2;
      end;
   
      /// Возвращает вектор, к каждому члену которого применена функция активации Синусода
      static function __sinusoid(const input: real): real;
      begin
        result := sin(input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Синусода
      static function __sinusoid_derivative(const input: real): real;
      begin
        result := cos(input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена функция активации Softplus
      static function __softexponential(const input: real): real;
      begin
        if softexponential_alpha < 0 then
          result := -ln(1-softexponential_alpha*(input+softexponential_alpha))/softexponential_alpha
        else if softexponential_alpha > 0 then 
          result := (exp(softexponential_alpha*input)-1)/softexponential_alpha+softexponential_alpha
        else
          result := input
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Softplus
      static function __softexponential_derivative(const input: real): real;
      begin
        if softexponential_derivative_alpha < 0 then
          result := 1/(1 - softexponential_derivative_alpha*(softexponential_derivative_alpha+input))
        else
          result := exp(softexponential_derivative_alpha*input);
      end;

      /// Возвращает вектор, к каждому члену которого применена функция активации Softplus
      static function __softplus(const input: real): real;
      begin
        result := ln(1+exp(input));
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Softplus
      static function __softplus_derivative(const input: real): real;
      begin
        result := 1/(1+exp(-input));
      end;
                          
      /// Возвращает вектор, к каждому члену которого применена функция активации Softsign
      static function __softsign(const input: real): real;
      begin
        result := input/(1+abs(input));
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Softsign
      static function __softsign_derivative(const input: real): real;
      begin
        result := 1/(1+abs(input))**2;
      end;
 
      /// Возвращает вектор, к каждому члену которого применена Квадратная радиальная базисная функция активации
      static function __sqrbf(const input: real): real;
      begin
        if abs(input) < 1 then
          result := 1 - input**2/2
        else if abs(input) > 2 then
          result := 0
        else
          result := (2-abs(input))**2/2;
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная Квадратной радиальной базисной функции активации
      static function __sqrbf_derivative(const input: real): real;
      begin
        if abs(input) < 1 then
          result := -input
        else if abs(input) > 2 then
          result := 0
        else
          result := input - 2*sign(input);
      end;
      /// Возвращает вектор, к каждому члену которого применена функция активации Линейный выпрямитель
      static function __srelu(const input: real): real;
      begin
      if input <= srelu_tl then
        result := srelu_tl + srelu_al*(input - srelu_tl)
      else if input >= srelu_tr then 
        result := srelu_tr + srelu_ar*(input - srelu_tr)
      else
        result := input; 
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Линейный выпрямитель
      static function __srelu_derivative(const input: real): real;
      begin
        if input <= srelu_derivative_tl then
          result := srelu_derivative_al
        else if input >= srelu_derivative_tr then
          result := srelu_derivative_ar
        else
          result := 1
      end;
      
      /// Возвращает вектор, к каждому члену которого применена функции активации Гиперболический тангенс
      static function __tanh(const input: real): real;
      begin
        result := System.Math.Tanh(input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Гиперболический тангенс
      static function __tanh_derivative(const input: real): real;
      begin
        result := 1-System.Math.Tanh(input)**2;
      end;

    public
      /// Возвращает вектор, к каждому члену которого применена функции активации Арктангенс
      static function arctan(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__arctan, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Арктангенс
      static function arctan_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__arctan_derivative, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена функция активации Ареасинус
      static function arsinh(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__arsinh, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Ареасинус
      static function arsinh_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__arsinh_derivative, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена Выгнутая тождественная функция активации
      static function bent_identity(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__bent_identity, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная Выгнутой тождественной функции активации
      static function bent_identity_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__bent_identity_derivative, input);
      end;
 
      /// Возвращает вектор, к каждому члену которого применена функция активации Хевисайда
      static function binary_step(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__binary_step, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Хевисайда
      static function binary_step_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__binary_step_derivative, input);
      end;

//      /// Возвращает вектор, к каждому члену которого применена функция активации Биполярный линейный выпрямитель
//      static function brelu(const input: neo.neo_array): neo.neo_array;
//      begin
//        result := new neo.neo_array(input.shape()[0], input.shape()[1]);
//        result.set_size(input.size);
//        {$omp parallel for}
//        for var index := 0 to input.shape()[0] do
//          if index mod 2 = 0 then
//            result[index] := __relu(input[index])
//          else
//            result[index] := -__relu(input[index]);
//      end;
      
//      /// Возвращает вектор, к каждому члену которого применена производная функции активации Биполярный линейный выпрямитель
//      static function brelu_derivative(const input: neo.neo_array): neo.neo_array;
//      begin
//        result := new neo.neo_array;
//        result.set_size(input.size);
//        {$omp parallel for}
//        for var index := 0 to input.size-1 do
//          if index mod 2 = 0 then
//            result[index] := __relu_derivative(input[index])
//          else
//            result[index] := __relu_derivative(-input[index]);
//      end;

      /// Возвращает вектор, к каждому члену которого применена Тождественная функция активации
      static function elu(const alpha: real): function(const input: neo.neo_array):neo.neo_array;
      begin
        functions.elu_alpha := alpha;
        result := _elu;
      end;

      /// Возвращает вектор, к каждому члену которого применена Гауссова функция активации
      static function _elu(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__elu, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена Тождественная функция активации
      static function elu_derivative(const alpha: real): function(const input: neo.neo_array):neo.neo_array;
      begin
        functions.elu_derivative_alpha := alpha;
        result := _elu_derivative;
      end;

      /// Возвращает вектор, к каждому члену которого применена Гауссова функция активации
      static function _elu_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__elu_derivative, input);
      end;
      

      static function gaussian(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__gaussian, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная Гауссовой функции активации
      static function gaussian_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__gaussian_derivative, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена Тождественная функция активации
      static function identity(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__identity, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная Тождественной функции активации
      static function identity_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__identity_derivative, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена Тождественная функция активации
      static function isru(const alpha: real): function(const input: neo.neo_array):neo.neo_array;
      begin
        functions.isru_alpha := alpha;
        result := _isru;
      end;

      /// Возвращает вектор, к каждому члену которого применена Гауссова функция активации
      static function _isru(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__isru, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена Тождественная функция активации
      static function isru_derivative(const alpha: real): function(const input: neo.neo_array):neo.neo_array;
      begin
        functions.isru_derivative_alpha := alpha;
        result := _isru_derivative;
      end;

      /// Возвращает вектор, к каждому члену которого применена Гауссова функция активации
      static function _isru_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__isru_derivative, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена Тождественная функция активации
      static function isrlu(const alpha: real): function(const input: neo.neo_array):neo.neo_array;
      begin
        functions.isrlu_alpha := alpha;
        result := _isrlu;
      end;

      /// Возвращает вектор, к каждому члену которого применена Гауссова функция активации
      static function _isrlu(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__isrlu, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена Тождественная функция активации
      static function isrlu_derivative(const alpha: real): function(const input: neo.neo_array):neo.neo_array;
      begin
        functions.isrlu_derivative_alpha := alpha;
        result := _isrlu_derivative;
      end;

      /// Возвращает вектор, к каждому члену которого применена Гауссова функция активации
      static function _isrlu_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__isrlu_derivative, input);
      end;
              
      /// Возвращает вектор, к каждому члену которого применена Тождественная функция активации
      static function plu(const alpha, c: real): function(const input: neo.neo_array):neo.neo_array;
      begin
        functions.plu_alpha := alpha;
        functions.plu_c := c;
        result := _plu;
      end;

      /// Возвращает вектор, к каждому члену которого применена Гауссова функция активации
      static function _plu(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__plu, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена Тождественная функция активации
      static function plu_derivative(const alpha, c: real): function(const input: neo.neo_array):neo.neo_array;
      begin
        functions.plu_derivative_alpha := alpha;
        functions.plu_derivative_c := c;
        result := _plu_derivative;
      end;

      /// Возвращает вектор, к каждому члену которого применена Гауссова функция активации
      static function _plu_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__plu_derivative, input);
      end;         
           
      /// Возвращает вектор, к каждому члену которого применена Тождественная функция активации
      static function prelu(const alpha: real): function(const input: neo.neo_array):neo.neo_array;
      begin
        functions.prelu_alpha := alpha;
        result := _prelu;
      end;

      /// Возвращает вектор, к каждому члену которого применена Гауссова функция активации
      static function _prelu(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__prelu, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена Тождественная функция активации
      static function prelu_derivative(const alpha: real): function(const input: neo.neo_array):neo.neo_array;
      begin
        functions.prelu_derivative_alpha := alpha;
        result := _prelu_derivative;
      end;

      /// Возвращает вектор, к каждому члену которого применена Гауссова функция активации
      static function _prelu_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__prelu_derivative, input);
      end;            
           
      /// Возвращает вектор, к каждому члену которого применена функция активации Линейный выпрямитель
      static function relu(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__relu, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Линейный выпрямитель
      static function relu_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__relu_derivative, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена функция активации Сигмоида
      static function sigmoid(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__sigmoid, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Сигмоида
      static function sigmoid_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__sigmoid_derivative, input);
      end;
                  
      /// Возвращает вектор, к каждому члену которого применена функция активации Кардинальный синус
      static function sinc(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__sinc, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Кардинальный синус
      static function sinc_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__sinc_derivative, input);
      end;
   
      /// Возвращает вектор, к каждому члену которого применена функция активации Синусода
      static function sinusoid(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__sinusoid, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Синусода
      static function sinusoid_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__sinusoid_derivative, input);
      end;
                       
      /// Возвращает вектор, к каждому члену которого применена Тождественная функция активации
      static function softexponential(const alpha: real): function(const input: neo.neo_array):neo.neo_array;
      begin
        functions.softexponential_alpha := alpha;
        result := _softexponential;
      end;

      /// Возвращает вектор, к каждому члену которого применена Гауссова функция активации
      static function _softexponential(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__softexponential, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена Тождественная функция активации
      static function softexponential_derivative(const alpha: real): function(const input: neo.neo_array):neo.neo_array;
      begin
        functions.softexponential_derivative_alpha := alpha;
        result := _softexponential_derivative;
      end;

      /// Возвращает вектор, к каждому члену которого применена Гауссова функция активации
      static function _softexponential_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__softexponential_derivative, input);
      end;            
           
//      /// Возвращает вектор, к каждому члену которого применена функция активации Softmax
//      static function softmax(const input: neo.neo_array): neo.neo_array;
//      begin
//        result := new neo.neo_array(input.shapes());
//        result.set_size(input.size);
//        {$omp parallel for}
//        for var index := 0 to input.size-1 do
//          result[index] := exp(input[index]);
//        var sum := result.sum();
//        result /= sum;
//      end;
//      
//      /// Возвращает вектор, к каждому члену которого применена производная функции активации Softmax
//      static function softmax_derivative(const input: neo.neo_array): neo.neo_array;
//      begin
//        result := softmax(input);
//        {$omp parallel for}
//        for var index := 0 to input.size-1 do
//          result[index] := result[index]*(1-result[index]);
//      end;

      /// Возвращает вектор, к каждому члену которого применена функция активации Softplus
      static function softplus(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__softplus, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Softplus
      static function softplus_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__softplus_derivative, input);
      end;
                          
      /// Возвращает вектор, к каждому члену которого применена функция активации Softsign
      static function softsign(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__softsign, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Softsign
      static function softsign_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__softsign_derivative, input);
      end;
 
      /// Возвращает вектор, к каждому члену которого применена Квадратная радиальная базисная функция активации
      static function sqrbf(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__sqrbf, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная Квадратной радиальной базисной функции активации
      static function sqrbf_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__sqrbf_derivative, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена Тождественная функция активации
      static function srelu(const al, ar, tl, tr: real): function(const input: neo.neo_array):neo.neo_array;
      begin
        functions.srelu_al := al;
        functions.srelu_ar := ar;
        functions.srelu_tl := tl;
        functions.srelu_tr := tr;
        result := _srelu;
      end;

      /// Возвращает вектор, к каждому члену которого применена Гауссова функция активации
      static function _srelu(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__srelu, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена Тождественная функция активации
      static function srelu_derivative(const al, ar, tl, tr: real): function(const input: neo.neo_array):neo.neo_array;
      begin
        functions.srelu_derivative_al := al;
        functions.srelu_derivative_ar := ar;
        functions.srelu_derivative_tl := tl;
        functions.srelu_derivative_tr := tr;
        result := _srelu_derivative;
      end;

      /// Возвращает вектор, к каждому члену которого применена Гауссова функция активации
      static function _srelu_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__srelu_derivative, input);
      end;  

      /// Возвращает вектор, к каждому члену которого применена функции активации Гиперболический тангенс
      static function tanh(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__tanh, input);
      end;
      
      /// Возвращает вектор, к каждому члену которого применена производная функции активации Гиперболический тангенс
      static function tanh_derivative(const input: neo.neo_array): neo.neo_array;
      begin
        result := neo.map(__tanh_derivative, input);
      end;
  end;
  
  loss_functions_type = function(const true_answer, received_answer: neo.neo_array): real;

  loss_functions = class
    public
      static function tmp_sqr(x: real): real;
      begin
        result := x**2;
      end;
    
      static function mse(const true_answer, received_answer: neo.neo_array): real;
      begin
        result := neo.map(tmp_sqr, (true_answer-received_answer)).sum()/received_answer.shapes()[0];
      end;

      static function root_mse(const true_answer, received_answer: neo.neo_array): real;
      begin
        result := System.Math.Sqrt(neo.map(tmp_sqr, (true_answer-received_answer)).sum()/received_answer.shapes()[0]);
      end;

  end;
//
//  Neuron = class
//    private
//      weights: neo.neo_array; 
//      input: neo.neo_array;
//      
//      /// Иницилизирует number_of_weights случайных весов в диапазоне [0, 1)
//      function initialize_weights(const number_of_weights: uint64): neo.neo_array;
//      var
//        tmp_result: array of real;
//      begin
//        tmp_result := new real[number_of_weights];
//        {$omp parallel for}
//        for var index := 0 to number_of_weights-1 do
//          tmp_result[index] := random(global_initializing_weights_range[0],
//                                      global_initializing_weights_range[1]);
//        result := new neo.neo_array(tmp_result);
//      end;
//      
//    public
//      /// Инициализирует новый экземпляр класса Neuron с number_of_inputs входов
//      constructor Create(const number_of_inputs: uint64);
//      begin
//        self.weights := initialize_weights(number_of_inputs);
//      end;
//      
//      /// Возвращает ненормализированный выход нейрона
//      function calculate(const input: neo.neo_array): real;
//      begin
//        self.input := input;
////        result := self.weights.dot(self.input);
//      end;
//
//      /// Возвращает вектор ошибок для предшестующих нейронов 
//      function backprop(const input: real): neo.neo_array;
//      begin
//        result := self.weights * input;
//      end;
//      
//      /// Изменяет веса с учетом ошибки delta
//      procedure adjust_weights(const delta: real);
//      begin
//        self.weights := self.weights + self.input * delta * global_alpha;
//      end;
//      
//      function ToString: string; override;
//      begin
//        result := 'Нейрон (Веса): ' + self.weights.ToString;
//      end;
//    end;
//
//  Layer = class
//    private
//      layer: array of Neuron;
//      
//    public
//      /// Инициализирует новый экземпляр класса Layer с number_of_neurons нейронов и number_of_weights входов для каждого
//      constructor Create(const number_of_neurons, number_of_weights: uint64);
//      begin
//        if number_of_neurons < 1 then
//          raise new System.ArgumentException('Размеры любого слоя ИНС должны быть больше 0');
//
//        self.layer := new Neuron[number_of_neurons];
//        {$omp parallel for}
//        for var index := 0 to number_of_neurons-1 do
//          self.layer[index] := new Neuron(number_of_weights);
//      end;
//      
//      /// Возвращает ненормализированный вектор выходных значений слоя
//      function calculate(const input: neo.neo_array): neo.neo_array;
//      begin
//        result := neo.map();
//        result.set_size(self.layer.Length);
//        {$omp parallel for}
//        for var index := 0 to self.layer.length-1 do
//          result[index] := self.layer[index].calculate(input); 
//      end;
//      
//      /// Возвращает вектор ошибок для предшестующих слоев 
//      function backprop(const input: neo.neo_array): neo.neo_array;
//      begin
//        result := self.layer[0].backprop(input[0]);
//        {$omp parallel for reduction(+:result)}
//        for var index := 1 to self.layer.Length-1 do
//          result += self.layer[index].backprop(input[index]);
//      end;
//      
//      /// Изменяет веса нейронов слоя с учетом ошибки delta
//      procedure adjust_weights(const delta: neo.neo_array);
//      begin
//        {$omp parallel for}
//        for var index := 0 to self.layer.Length-1 do
//          self.layer[index].adjust_weights(delta[index]);
//      end;
//      
//      function ToString: string; override;
//      begin
//        result := 'Слой: ';
//        for var index := 0 to layer.Length-2 do
//          result += layer[index].ToString + ' | ';
//        result += layer[layer.Length-1].ToString;
//      end;
//    end;
//
//  Neural_Network = class
//    private
//      seed: integer;
//    
//      neural_network: array of Layer;
//      topology: neo.neo_array;
//      number_of_layers: uint64;
//      activation_functions: array of function(const input: neo.neo_array): neo.neo_array;
//      activation_functions_derivatives: array of nntk.functions_type;
//      loss_function: loss_functions_type;
//      batch_size: uint64;
//
//      /// Обучает нейронную сеть на входных данных input_data и выходных данных output_data number_of_epoch эпох 
//      procedure __train(const train_input_data: List<neo.neo_array>; 
//                        const train_output_data: List<neo.neo_array>;
//                        const test_input_data: List<neo.neo_array>;
//                        const test_output_data: List<neo.neo_array>;
//                        const number_of_epoch: uint64);
//      var
//        deltas: array of neo.neo_array;
//        layers: array of neo.neo_array;
//        mask: array of neo.neo_array;
//        error: real;
//        test_error: real;
//      begin
//        deltas := new neo.neo_array[self.number_of_layers-1]; 
//        {$omp parallel for}
//        for var index := 0 to self.number_of_layers-2 do
//          begin
//          deltas[index] := new neo.neo_array;
//          deltas[index].set_size(trunc(topology[self.number_of_layers-index-1]));
//          end;
//        layers := new neo.neo_array[self.number_of_layers];
//        mask := new neo.neo_array[self.number_of_layers-1];
//
//        for var epoch := 1 to number_of_epoch do
//        begin 
//          error := 0.0;
//          for var index := 0 to train_input_data.count-1 do
//            begin
//              layers[0] := train_input_data[index];
//              for var i := 0 to self.number_of_layers-2 do
//                layers[i+1] := self.activation_functions[i](self.neural_network[i].calculate(layers[i]));
//              
//              {$omp parallel for}
//              for var i := 1 to self.number_of_layers-2 do
//              begin
//                mask[i-1] := get_dropout_mask(layers[i].size());
//                layers[i] *= mask[i-1] * (1/(1-global_dropout_probability));
//              end; 
//                            
//              error += self.loss_function(train_output_data[index], layers.Last);
//
//              deltas[0] += train_output_data[index]-layers.last();
//              for var i := 1 to self.number_of_layers-2 do
//                deltas[i] += self.neural_network[self.number_of_layers-i-1].backprop(deltas[i-1])
//                           * self.activation_functions_derivatives[self.number_of_layers-i-1](layers[self.number_of_layers-i-1])
//                           * mask[self.number_of_layers-i-2];
//
//              if ((epoch-1)*train_input_data.Count+index+1) mod self.batch_size = 0 then
//                {$omp parallel for}
//                for var i := 0 to self.number_of_layers-2 do
//                begin
//                  self.neural_network[i].adjust_weights(deltas[self.number_of_layers-2-i]/self.batch_size);
//                  deltas[self.number_of_layers-2-i] := new neo.neo_array;
//                  deltas[self.number_of_layers-2-i].set_size(trunc(topology[i+1]));
//                  end;
//            end;
//          if (epoch) mod 10 = 0 then
//          begin
//            for var index := 0 to test_input_data.count-1 do
//              begin
//              layers[0] := test_input_data[index];
//              for var i := 0 to self.number_of_layers-2 do
//                layers[i+1] := self.activation_functions[i](self.neural_network[i].calculate(layers[i]));
//              test_error += self.loss_function(test_output_data[index], layers.Last);
//              end;
//              
//            println(format('I:{0} Train-Error:{1,7:f5}   Test-Error:{2,7:f5}', 
//                           epoch, error/train_input_data.count, test_error/test_input_data.count));
//            test_error := 0.0;
//            end;
//          end;
//      end;
//      
//    public
//      /// Инициализирует объект класса Neural_Network с заданной топологией
//      constructor Create(const neural_network_topology: neo.neo_array;
//                         initializing_weights_range: System.Tuple<real, real> := (-1.0, 1.0);
//                         activation_functions: array of nntk.functions_type := nil;
//                         activation_functions_derivatives: array of nntk.functions_type := nil;
//                         seed: integer := 0);
//      begin
//        Randomize(seed);
//        self.topology := neural_network_topology;
//                self.number_of_layers := neural_network_topology.size();
//        if self.number_of_layers < 2 then
//          raise new System.ArgumentException('Кол-во слоев ИНС должно быть больше 1');
//        
//        global_initializing_weights_range := initializing_weights_range;
//        
//        if activation_functions = nil then
//          begin
//          Setlength(self.activation_functions, self.number_of_layers-1);
//          for var index := 0 to self.number_of_layers-2 do
//            self.activation_functions[index] := nntk.Functions.sigmoid;
//          end
//        else if activation_functions.Length <> self.number_of_layers-1 then
//          raise new System.ArgumentException('Кол-во функций активации и кол-во слоев ИНС данных должны совпадать')
//        else
//          begin
//            Setlength(self.activation_functions, self.number_of_layers-1);
//            for var index := 0 to self.number_of_layers-2 do
//              self.activation_functions[index] := activation_functions[index];
//          end;
//          
//        if activation_functions_derivatives = nil then
//          begin
//          Setlength(activation_functions_derivatives, self.number_of_layers-1);
//          for var index := 0 to self.number_of_layers-2 do
//            self.activation_functions_derivatives[index] := nntk.Functions.sigmoid_derivative;
//          end
//        else if activation_functions_derivatives.length <> self.number_of_layers-1 then
//          raise new System.ArgumentException('Кол-во производных функций активации и кол-во слоев ИНС данных должны совпадать')
//        else
//          begin
//            Setlength(self.activation_functions_derivatives, self.number_of_layers-1);
//            for var index := 0 to self.number_of_layers-2 do
//              self.activation_functions_derivatives[index] := activation_functions_derivatives[index];
//          end;
//          
//        self.neural_network := new Layer[self.number_of_layers-1];
//        {$omp parallel for}
//        for var index := 1 to self.number_of_layers-1 do
//          self.neural_network[index-1] := new Layer(trunc(neural_network_topology[index]),
//                                                  trunc(neural_network_topology[index-1]));
//        end;
// 
//      /// Обучает нейронную сеть на входных данных input_data и выходных данных output_data number_of_epoch эпох
//      /// Необязательные параметры:
//      /// Коэффициент обучаемости alpha (по умолчанию 0.01)
//      /// Функция активации activation_function (по умолчанию nntk.functions.relu) 
//      /// Производная фукции активации activation_function_derivative (по умолчанию nntk.functions.relu_derivative)
//      /// Вероятность прореживания dropout_probability (по умолчанию 0.0 - Прореживание не проводится)
//      /// Размер пакета разности весов batch_size (по умолчанию 1 - Обучение происходит на каждом отдельном примере)
//      procedure train(const train_input_data: array of neo.neo_array; 
//                      const train_output_data: array of neo.neo_array;
//                      const test_input_data: array of neo.neo_array; 
//                      const test_output_data: array of neo.neo_array;
//                      const number_of_epoch: uint64;
//                      alpha: real := 0.01;
//                      loss_function: loss_functions_type := nntk.loss_functions.mse;
//                      dropout_probability: real := 0.0;
//                      batch_size: uint64 := 1);
//                      
//      begin
//        if train_input_data.Length <> train_output_data.Length then
//          raise new System.ArgumentException('Размеры обучающей выборки для входных и выходных данных должны совпадать');
//        global_alpha := alpha;
//        self.loss_function := loss_function;
//        if (0 > dropout_probability) or (dropout_probability >= 1) then
//          raise new System.ArgumentException('Вероятность прореживания узлов нейронной сети должна принадлежать [0, 1)');
//        global_dropout_probability := dropout_probability;       
//        if batch_size < 1 then
//          raise new System.ArgumentException('Размер пакетов разности весов должен быть больше нуля');
//        self.batch_size := batch_size;        
//        __train(new List<neo.neo_array>(train_input_data), 
//                new List<neo.neo_array>(train_output_data), 
//                new List<neo.neo_array>(test_input_data), 
//                new List<neo.neo_array>(test_output_data), 
//                number_of_epoch);
//
//      end;
//
//      /// Возвращает результат работы нейронной сети для входных данных input_data
//      function run(const input_data: neo.neo_array): neo.neo_array;
//      begin
//      if input_data.size() <> self.topology[0] then
//          raise new System.ArgumentException('Размеры первого слоя ИНС и входных данных должны совпадать');  
//      var layers := new neo.neo_array[self.number_of_layers];
//      layers[0] := input_data;
//      for var i := 0 to self.number_of_layers-2 do
//        layers[i+1] := self.activation_functions[i](self.neural_network[i].calculate(layers[i]));
//      result := layers[self.number_of_layers-1];
//      end;
//      
//      /// Возвращает модель нейронной сети в виде функции
//      function get_model(): function (input_data: neo.neo_array): neo.neo_array;
//      begin
//        result := self.run;
//      end;
//      
//      /// Возвращает модель вектор прореживания
//      function get_dropout_mask(const size: uint64): neo.neo_array;
//      begin
//        result := new neo.neo_array;
//        result.set_size(size);
//        {$omp parallel for}
//        for var index := 0 to size-1 do
//          result[index] := random() < global_dropout_probability? 0: 1;
//      end;
//  end;
//  
  Tensor = class
    private
      id: integer;
      static id_counter := 0;
      data: neo.neo_array;
      autograde: boolean;
      creators: array of Tensor := nil;
      children: Dictionary<integer, integer>;
      creation_op: string;
      grad: Tensor := nil;
      
    public
    
      constructor Create(data: neo.neo_array; 
                         autograde: boolean := False;
                         creators: array of Tensor := nil; 
                         creation_op: string := nil;
                         id: integer := -1);
      begin
        self.data := data;
        self.autograde := autograde;
        self.creators := creators;
        self.creation_op := creation_op;
        self.children := new Dictionary<integer, integer>;
        
        if id = -1 then
          begin
          self.id := id_counter;
          id_counter += 1;
          end;
          
        if self.creators <> nil then
          for var index := 0 to self.creators.Length-1 do
            if self.creators[index].children.ContainsKey(self.id) then
              self.creators[index].children[self.id] += 1
            else
              self.creators[index].children[self.id] := 1;
      end;
      
      procedure backward(grad: Tensor := nil; grad_origin: Tensor := nil);
      begin
        if self.autograde then
        begin
          if grad_origin <> nil then
            if self.children[grad_origin.id] = 0 then
              raise new Exception('cannot backprop more than once')
            else
              self.children[grad_origin.id] -= 1;
          if self.grad <> nil then
            self.grad += grad
          else 
            self.grad := grad;
          if (self.creators <> nil) and ((self.children.Values.All(x->x=0)) or (grad_origin = nil)) then
            if self.creation_op = 'add' then
            begin
              self.creators[0].backward(self.grad, self);
              self.creators[1].backward(self.grad, self);
            end
            else if self.creation_op = 'neg' then
              self.creators[0].backward(-self.grad, self)
            else if self.creation_op = 'sub' then
              begin
              self.creators[0].backward(self.grad, self); 
              self.creators[1].backward(-self.grad, self);
              end
            else if self.creation_op = 'mul' then 
              begin
              self.creators[0].backward(self.grad * self.creators[0], self);
              self.creators[1].backward(self.grad * self.creators[1], self);
              end
//            else if(self.creation_op == "mm"): 
//              act = self .creators [0] 
//              weights = self .creators [1]
//              new = self.grad.mm(weights.transpose()) act.backward(new) 
//              new = self.grad.transpose().mm(act).transpose() 
//              weights.backward(new)
//            else if(self.creation_op == "transpose"): 
//              self.creators[0].backward(self.grad.transpose())
            else if self.creation_op.StartsWith('sum') then 
              begin
                var dim := integer.Parse(self.creation_op.Split('_')[1]);
                var ds := self.creators[0].data.shapes[dim];
//                self.creators[0].backward(self.grad.expand(dim,ds));
              end;
//            if("expand" in self.creation_op): 
//              dim = int(self.creation_op.split("_")[1]) 
//              self.creators[0].backward(self.grad.sum(dim))
        end;
      end;
      
      static function operator+(self_tensor, other_tensor: Tensor): Tensor;
      begin
        if self_tensor.autograde and other_tensor.autograde then
        begin
          var tmp_creators: array of Tensor := (self_tensor, other_tensor);
          result := new Tensor(self_tensor.data + other_tensor.data, True, tmp_creators, 'add');
        end
        else
          result := new Tensor(self_tensor.data + other_tensor.data, False, nil, 'add');
      end;
      static procedure operator+=(var self_tensor, other_tensor: Tensor);
      begin
        self_tensor := self_tensor + other_tensor;
      end;
      
      static function operator-(self_tensor: Tensor): Tensor;
      begin
        if self_tensor.autograde then
        begin
          var tmp_creator: array of tensor := (self_tensor);
          result := new Tensor(self_tensor.data * (-1), True, tmp_creator, 'neg')
        end
        else
          result := new Tensor(self_tensor.data * (-1), False, nil, 'neg');
      end;
      
      static function operator-(self_tensor, other_tensor: Tensor): Tensor;
      begin
        if self_tensor.autograde and other_tensor.autograde then
        begin
          var tmp_creators: array of Tensor := (self_tensor, other_tensor);
          result := new Tensor(self_tensor.data - other_tensor.data, True, tmp_creators, 'sub');
        end
        else
          result := new Tensor(self_tensor.data - other_tensor.data, False, nil, 'sub');
      end;
      static procedure operator-=(var self_tensor, other_tensor: Tensor);
      begin
        self_tensor := self_tensor - other_tensor;
      end;
      
      static function operator*(self_tensor, other_tensor: Tensor): Tensor;
      begin
        if self_tensor.autograde and other_tensor.autograde then
        begin
          var tmp_creators: array of Tensor := (self_tensor, other_tensor);
          result := new Tensor(self_tensor.data * other_tensor.data, True, tmp_creators, 'mul');
        end
        else
          result := new Tensor(self_tensor.data * other_tensor.data, False, nil, 'mul');
      end;
      static procedure operator*=(var self_tensor, other_tensor: Tensor);
      begin
        self_tensor := self_tensor * other_tensor;
      end;
      
      function sum(): Tensor;
      begin
        var sum_array: array of real := (self.data.sum());
        if self.autograde then
        begin
          var tmp_creator: array of Tensor := (self);
          result := new Tensor(new neo.neo_array(sum_array), True, tmp_creator, 'sum_0')
        end
        else
          result := new Tensor(new neo.neo_array(sum_array), False, nil, 'sum_0');
      end;
      
//      function expand(dim, copies: integer): Tensor;
//      begin
//        var trans_cmd := [0..self.data.shape.length]; 
//        trans_cmd.insert(dim,len(self.data.shape)); 
//        var new_shape := list(self.data.shape) + [copies]; 
//        var new_data := self.data.repeat(copies).reshape(new_shape); 
//        new_data := new_data.transpose(trans_cmd);
//        if self.autograd then 
//        begin
//          var tmp_creator: array of Tensor := (self);
//          result := Tensor(new_data, True, tmp_creator, 'expand_'+str(dim));
//          end
//        else
//          result := Tensor(new_data, False, nil, 'expand_'+str(dim));
//      end;
      
//      function transpose(): Tensor; 
//        begin
//        if self.autograde then
//        begin
//          var tmp_creator: array of Tensor := (self);
//          result := new Tensor(self.data.transpose(), True, tmp_creator, creation_op='transpose') 
//          end
//        else
//          result := new Tensor(self.data.transpose(), False, nil, 'transpose')
//        end;
        
//      function mm(other_tensor: Tensor): Tensor; 
//        begin
//        if self.autograde then
//          begin
//          var tmp_creators: array of Tensor := (self, other_tensor);
//          result := new Tensor(self.data.dot(other_tensor.data), True, tmp_creators, 'mm')
//          end
//        else
//          result := new Tensor(self.data.dot(other_tensor.data), False, nil, 'mm')
//        end;
      
      function ToString: string; override;
      begin
        result += self.data.ToString + ': ';
        for var index := 0 to self.data.shapes()[0] do
          result += self.data[0, index] + ', ';
        result += self.data[0, self.data.shapes()[0]-1].ToString + ' (';

        if self.creators <> nil then
          for var index := 0 to self.creators.Length-1 do
            result += self.creators[index].ToString + ', ';
        result += ')('+self.creation_op+')(';
        if self.grad <> nil then
          for var index := 0 to self.grad.data.shapes()[0]-1 do
            result += self.grad.data[0, index].ToString + ', ';
        result += ')';
      end;
  end;
  
end.