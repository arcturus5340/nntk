unit vector_math;

type 
  Vector<T> = class
    private
      vector: array of T;
    public
      property Element[index: integer]: T read self.vector[index] 
                                          write self.vector[index] := value;
                                          default;
                                          
      constructor Create(params list_of_values: array of T);
      begin
        self.vector := list_of_values;
      end;
      constructor Create(list_of_values: List<T>);
      begin
        self.vector := list_of_values.ToArray;
      end;
      
      function dot(other_vector: Vector<real>): real;
      begin
        result := 0.0;
        for var index := 0 to self.vector.Length-1 do
          begin
            var operand_1 := real.Parse(self.vector[index].ToString);
            var operand_2 := other_vector[index];
            result += operand_1 * operand_2;
          end;
      end;
      
      static function operator-(self_vector, other_vector: Vector<T>): Vector<real>;
      begin
        var tmp_result := new List<real>;
        for var index := 0 to self_vector.size-1 do
        begin
          var operand_1 := real.Parse(self_vector[index].ToString);
          var operand_2 := real.Parse(other_vector[index].ToString);
          tmp_result.add(operand_1 - operand_2);
        end;
        result := new Vector<real>(tmp_result);
      end;

      static function operator+(self_vector, other_vector: Vector<T>): Vector<real>;
      begin
        var tmp_result := new List<real>;
        for var index := 0 to self_vector.size-1 do
        begin
          var operand_1 := real.Parse(self_vector[index].ToString);
          var operand_2 := real.Parse(other_vector[index].ToString);
          tmp_result.add(operand_1 + operand_2);
        end;
        result := new Vector<real>(tmp_result);
      end;  
      static function operator+(self_vector: Vector<T>; other_operand: T): Vector<real>;
      begin
        var tmp_result := new List<real>;
        var operand_2 := real.Parse(other_operand.ToString);
        for var index := 0 to self_vector.size-1 do
        begin
          var operand_1 := real.Parse(self_vector[index].ToString);
          tmp_result.add(operand_1 + operand_2);
        end;
        result := new Vector<real>(tmp_result);
      end;
      
      static function operator*(self_vector, other_vector: Vector<T>): Vector<real>;
      begin
        var tmp_result := new List<real>;
        if other_vector.size() = 1 then
        begin
          var operand_2 := real.Parse(other_vector[0].ToString);
          for var index := 0 to self_vector.size-1 do
          begin
            var operand_1 := real.Parse(self_vector[index].ToString);
            tmp_result.add(operand_1 * operand_2);
          end;
        end
        else
          for var index := 0 to self_vector.size-1 do
          begin
            var operand_1 := real.Parse(self_vector[index].ToString);
            var operand_2 := real.Parse(other_vector[index].ToString);
            tmp_result.add(operand_1 * operand_2);
          end;
        result := new Vector<real>(tmp_result);
      end;      
      static function operator*(self_vector: Vector<T>; other_operand: T): Vector<real>;
      begin
        var tmp_result := new List<real>;
        var operand_2 := real.Parse(other_operand.ToString);
        for var index := 0 to self_vector.size-1 do
        begin
          var operand_1 := real.Parse(self_vector[index].ToString);
          tmp_result.add(operand_1 * operand_2);
        end;
        result := new Vector<real>(tmp_result);
      end;

      static function operator**(self_vector: Vector<T>; number: T): Vector<real>;
      begin
        var tmp_result := new List<real>;
        var operand_2 := real.Parse(number.ToString);
        for var index := 0 to self_vector.size-1 do
        begin
          var operand_1 := real.Parse(self_vector[index].ToString);
          tmp_result.add(operand_1 ** operand_2);
        end;
        result := new Vector<real>(tmp_result);
      end;
         
      function back(): T;
      begin
        result := self.vector[self.vector.Length-1];
      end;
      
      procedure push_back(x: T);
      begin
        self.vector := self.vector.append(x).ToArray;  
      end;
      
      function size(): integer;
      begin
        result := self.vector.Length;
      end;
      
      function ToString: string; override;
      begin
        result := 'Vector(';
        for var index := 0 to self.vector.Length-2 do
          result += self.vector[index].ToString + ', ';
        result += self.vector[self.vector.Length-1].ToString + ')';
      end;
  end;

end.
