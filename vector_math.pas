/// Модуль для работы с математическими векторами
unit vector_math;

type 
  Vector = class
    private
      coordinates: array of single; 
      
      static function __add(const self_vector, other_vector: Vector): Vector;
      var 
        tmp_result: array of single;
      begin
        tmp_result := new single[self_vector.size()];
        tmp_result.Initialize();
        {$omp parallel for}
        for var index := 0 to self_vector.size-1 do
          tmp_result[index] := self_vector[index] + other_vector[index];
        result := new Vector(tmp_result);
      end;
      
      static function __sub(const self_vector, other_vector: Vector): Vector;
      var 
        tmp_result: array of single;
      begin
        tmp_result := new single[self_vector.size()];
        tmp_result.Initialize();
        {$omp parallel for}
        for var index := 0 to self_vector.size-1 do
          tmp_result[index] := self_vector[index] - other_vector[index];
        result := new Vector(tmp_result);
      end;
      
      static function __mul(const self_vector, other_vector: Vector): Vector;
      var
        tmp_result: array of single;
      begin
        tmp_result := new single[self_vector.size()];
        tmp_result.Initialize();
        if other_vector.size() = 1 then
          begin
          if other_vector[0] = 0.0 then
            result := new Vector(tmp_result)
          else
            {$omp parallel for}
            for var index := 0 to self_vector.size-1 do
              tmp_result[index] := self_vector[index] * other_vector[0]
          end
        else
          {$omp parallel for}
          for var index := 0 to self_vector.size-1 do
            tmp_result[index] := self_vector[index] * other_vector[index];
        result := new Vector(tmp_result);
      end;
      
      static function __div(const self_vector, other_vector: Vector): Vector;
      var
        tmp_result: array of single;
      begin
        tmp_result := new single[self_vector.size()];
        tmp_result.Initialize();
        {$omp parallel for}
        for var index := 0 to self_vector.size-1 do
          tmp_result[index] := self_vector[index] / other_vector[0];
        result := new Vector(tmp_result);
      end;
      
      static function __pow(const self_vector: Vector; exponent: byte): Vector;
      var 
        tmp_result: array of single;
      begin
        tmp_result := new single[self_vector.size()];
        tmp_result.Initialize();
        {$omp parallel for}
        for var index := 0 to self_vector.size-1 do
          tmp_result[index] := self_vector[index] ** exponent;
        result := new Vector(tmp_result);
      end;
    
    public
      property Element[index: uint64]: single read self.coordinates[index] 
                                               write self.coordinates[index] := value;
                                               default;
      /// Инициализирует новый экземпляр класса Vector, содержащий элементы массива array_of_values
      constructor Create(params array_of_values: array of single);
      begin
        self.coordinates := array_of_values;
      end;
      /// Инициализирует новый экземпляр класса Vector, содержащий элементы листа list_of_values
      constructor Create(const list_of_values: List<single>);
      begin
        self.coordinates := list_of_values.ToArray;
      end;

      /// Возвращает скалярное произведение двух векторов
      function dot(const other_vector: Vector): single;
      begin
        if self.coordinates.Length <> other_vector.size then
          raise new System.ArithmeticException('Размеры векторов не совпадают');
        {$omp parallel for reduction(+:result)}
        for var index := 0 to self.coordinates.Length-1 do
          begin
            var operand_1 := self.coordinates[index];
            var operand_2 := other_vector[index];
            result += operand_1 * operand_2;
          end;
      end;
      
      /// Возвращает сумму элементов вектора
      function sum(): single;
      begin
        {$omp parallel for reduction(+:result)}
        for var index := 0 to self.coordinates.Length-1 do
          result += self.coordinates[index];
      end;
      
      static function operator+(const self_vector, other_vector: Vector): Vector;
      begin
        if self_vector.size <> other_vector.size then
          raise new System.ArithmeticException('Размеры векторов не совпадают');
        result := __add(self_vector, other_vector);
      end;  
      
      static procedure operator+=(var self_vector: Vector; const other_vector: Vector);
      begin
        if self_vector.size <> other_vector.size then
          raise new System.ArithmeticException('Размеры векторов не совпадают');
        self_vector := __add(self_vector, other_Vector);
      end;  
     
      static function operator-(const self_vector, other_vector: Vector): Vector;
      begin
        if self_vector.size <> other_vector.size then
          raise new System.ArithmeticException('Размеры векторов не совпадают');
        result := __sub(self_vector, other_vector);
      end;
      
      static procedure operator-=(var self_vector: Vector; const other_vector: Vector);
      begin
        if self_vector.size <> other_vector.size then
          raise new System.ArithmeticException('Размеры векторов не совпадают');
        self_vector := __sub(self_vector, other_Vector);
      end;  
      
      static function operator*(const self_vector, other_vector: Vector): Vector;
      begin
        if self_vector.size <> other_vector.size then
          raise new System.ArithmeticException('Размеры векторов не совпадают');
        result := __mul(self_vector, other_vector);
      end;      
      
      static procedure operator*=(var self_vector: Vector; const other_vector: Vector);
      begin
        if self_vector.size <> other_vector.size then
          raise new System.ArithmeticException('Размеры векторов не совпадают');
        self_vector := __mul(self_vector, other_Vector);
      end;  
      
      static function operator*(const self_vector: Vector; other_operand: single): Vector;
      begin
        result := __mul(self_vector, new Vector(other_operand));
      end;
      
      static procedure operator*=(var self_vector: Vector; const other_operand: single);
      begin
        self_vector := __mul(self_vector, new Vector(other_operand));
      end;  

      static function operator/(const self_vector: Vector; other_operand: single): Vector;
      begin
        result := __mul(self_vector, new Vector(other_operand));
      end;
      
      static procedure operator/=(var self_vector: Vector; const other_operand: single);
      begin
        self_vector := __div(self_vector, new Vector(other_operand));
      end;

      static function operator**(const self_vector: Vector; exponent: byte): Vector;
      begin
        result := __pow(self_vector, exponent);
      end;
      
      /// Возвращает последний элемент вектора
      function back(): single;
      begin
        result := self.coordinates[self.coordinates.Length-1];
      end;
      
      /// Возвращает кол-во элементов x в векторе
      function count(const x: single): uint64;
      begin
        result := self.coordinates.Count(val -> val=x);
      end;
      
      /// Помещает элемент x в конец вектора
      procedure push_back(const x: single);
      begin
        self.coordinates := self.coordinates.Append(x).ToArray();  
      end;
      
      /// Устанавливает размерность вектора в x
      procedure set_size(const x: uint64);
      begin
        SetLength(self.coordinates, x);
        self.coordinates.Initialize();
      end;
      
      /// Возвращает размерность вектора
      function size(): uint64;
      begin
        result := self.coordinates.Length;
      end;
      
      function ToString: string; override;
      begin
        result := 'Vector(';
        for var index := 0 to self.coordinates.Length-2 do
          result += self.coordinates[index].ToString + ', ';
        result += ((self.coordinates.Length>0)?self.coordinates[self.coordinates.Length-1].ToString:'') + ')';
      end;
  end;
end.