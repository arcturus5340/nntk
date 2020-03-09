unit vector_math;

type 
  Vector = class
    private
    public
      coordinates: array of real;
    
      property Element[index: integer]: real read self.coordinates[index] 
                                             write self.coordinates[index] := value;
                                             default;
                                          
      constructor Create(params list_of_values: array of real);
      begin
        self.coordinates := list_of_values;
      end;
      constructor Create(list_of_values: List<real>);
      begin
        self.coordinates := list_of_values.ToArray;
      end;
      
      function dot(other_coordinates: Vector): real;
      begin
        result := 0.0;
        for var index := 0 to self.coordinates.Length-1 do
          begin
            var operand_1 := real.Parse(self.coordinates[index].ToString);
            var operand_2 := other_coordinates[index];
            result += operand_1 * operand_2;
          end;
      end;
      
      function sum(): real;
      begin
        result := 0.0;
        for var index := 0 to self.coordinates.Length-1 do
          begin
            var operand_1 := real.Parse(self.coordinates[index].ToString);
            result += operand_1;
          end;
      end;
      
      static function operator-(self_coordinates, other_coordinates: Vector): Vector;
      begin
        var tmp_result := new List<real>;
        for var index := 0 to self_coordinates.size-1 do
        begin
          var operand_1 := real.Parse(self_coordinates[index].ToString);
          var operand_2 := real.Parse(other_coordinates[index].ToString);
          tmp_result.add(operand_1 - operand_2);
        end;
        result := new Vector(tmp_result);
      end;

      static function operator+(self_coordinates, other_coordinates: Vector): Vector;
      begin
        var tmp_result := new List<real>;
        for var index := 0 to self_coordinates.size-1 do
        begin
          var operand_1 := real.Parse(self_coordinates[index].ToString);
          var operand_2 := real.Parse(other_coordinates[index].ToString);
          tmp_result.add(operand_1 + operand_2);
        end;
        result := new Vector(tmp_result);
      end;  
      static function operator+(self_coordinates: Vector; other_operand: real): Vector;
      begin
        var tmp_result := new List<real>;
        var operand_2 := real.Parse(other_operand.ToString);
        for var index := 0 to self_coordinates.size-1 do
        begin
          var operand_1 := real.Parse(self_coordinates[index].ToString);
          tmp_result.add(operand_1 + operand_2);
        end;
        result := new Vector(tmp_result);
      end;
      
      static function operator*(self_coordinates, other_coordinates: Vector): Vector;
      begin
        var tmp_result := new List<real>(other_coordinates.size());
        if other_coordinates.size() = 1 then
        begin
          var operand_2 := real.Parse(other_coordinates[0].ToString);
          for var index := 0 to self_coordinates.size-1 do
          begin
            var operand_1 := real.Parse(self_coordinates[index].ToString);
            tmp_result.add(operand_1 * operand_2);
          end;
        end
        else
          for var index := 0 to self_coordinates.size-1 do
          begin
            var operand_1 := real.Parse(self_coordinates[index].ToString);
            var operand_2 := real.Parse(other_coordinates[index].ToString);
            tmp_result.add(operand_1 * operand_2);
          end;
        result := new Vector(tmp_result);
      end;      
      static function operator*(self_coordinates: Vector; other_operand: real): Vector;
      begin
        var tmp_result := new List<real>;
        var operand_2 := real.Parse(other_operand.ToString);
        for var index := 0 to self_coordinates.size-1 do
        begin
          var operand_1 := real.Parse(self_coordinates[index].ToString);
          tmp_result.add(operand_1 * operand_2);
        end;
        result := new Vector(tmp_result);
      end;

      static function operator**(self_coordinates: Vector; number: real): Vector;
      begin
        var tmp_result := new List<real>;
        var operand_2 := real.Parse(number.ToString);
        for var index := 0 to self_coordinates.size-1 do
        begin
          var operand_1 := real.Parse(self_coordinates[index].ToString);
          tmp_result.add(operand_1 ** operand_2);
        end;
        result := new Vector(tmp_result);
      end;
         
      function back(): real;
      begin
        result := self.coordinates[self.coordinates.Length-1];
      end;
      
      procedure push_back(x: real);
      begin
        self.coordinates := self.coordinates.append(x).ToArray;  
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
