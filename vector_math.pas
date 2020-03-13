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
      property Element[index: integer]: single read self.coordinates[index] 
                                               write self.coordinates[index] := value;
                                               default;
                                          
      constructor Create(params list_of_values: array of single);
      begin
        self.coordinates := list_of_values;
      end;
      constructor Create(const list_of_values: List<single>);
      begin
        self.coordinates := list_of_values.ToArray;
      end;
      
      function dot(const other_vector: Vector): single;
      begin
        {$omp parallel for reduction(+:result)}
        for var index := 0 to self.coordinates.Length-1 do
          begin
            var operand_1 := self.coordinates[index];
            var operand_2 := other_vector[index];
            result += operand_1 * operand_2;
          end;
      end;
      
      function sum(): single;
      begin
        {$omp parallel for reduction(+:result)}
        for var index := 0 to self.coordinates.Length-1 do
          result += self.coordinates[index];
      end;
      
      static function operator+(const self_vector, other_vector: Vector): Vector;
      begin
        result := __add(self_vector, other_vector);
      end;  
     
      static function operator-(const self_vector, other_vector: Vector): Vector;
      begin
        result := __sub(self_vector, other_vector);
      end;
      
      static function operator*(const self_vector, other_vector: Vector): Vector;
      begin
        result := __mul(self_vector, other_vector);
      end;      
      
      static function operator*(const self_vector: Vector; other_operand: single): Vector;
      begin
        result := __mul(self_vector, new Vector(other_operand));
      end;

      static function operator**(const self_vector: Vector; exponent: integer): Vector;
      begin
        result := __pow(self_vector, exponent);
      end;
         
      function back(): single;
      begin
        result := self.coordinates[self.coordinates.Length-1];
      end;
      
      function count(const x: single): integer;
      begin
        result := self.coordinates.Count(val -> val=x);
      end;
      
      procedure push_back(const x: single);
      begin
        self.coordinates := self.coordinates.Append(x).ToArray();  
      end;
      
      procedure set_size(const x: integer);
      begin
        SetLength(self.coordinates, x);
        self.coordinates.Initialize();
      end;
      
      function size(): integer;
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
