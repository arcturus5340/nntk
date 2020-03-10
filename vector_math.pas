﻿unit vector_math;

type 
  Vector = class
    private
      coordinates: array of real; 
      
      static function __add(self_vector, other_vector: Vector): Vector;
      var 
        tmp_result: array of real;
      begin
        tmp_result := new real[self_vector.size()];
        tmp_result.Initialize();
        for var index := 0 to self_vector.size-1 do
        begin
          var operand_1 := self_vector[index];
          var operand_2 := other_vector[index];
          tmp_result[index] := operand_1 + operand_2;
        end;
        result := new Vector(tmp_result);
      end;
      
      static function __sub(self_vector, other_vector: Vector): Vector;
      var 
        tmp_result: array of real;
      begin
        tmp_result := new real[self_vector.size()];
        tmp_result.Initialize();
        for var index := 0 to self_vector.size-1 do
        begin
          var operand_1 := self_vector[index];
          var operand_2 := other_vector[index];
          tmp_result[index] := operand_1 - operand_2;
        end;
        result := new Vector(tmp_result);
      end;
      
      static function __mul(self_vector, other_vector: Vector): Vector;
      var
        tmp_result: array of real;
      begin
        tmp_result := new real[self_vector.size()];
        tmp_result.Initialize();
        if other_vector.size() = 1 then
        begin
          var operand_2 := other_vector[0];
          for var index := 0 to self_vector.size-1 do
          begin
            var operand_1 := self_vector[index];
            tmp_result[index] := operand_1 * operand_2;
          end;
        end
        else
          for var index := 0 to self_vector.size-1 do
          begin
            var operand_1 := self_vector[index];
            var operand_2 := other_vector[index];
            tmp_result[index] := operand_1 * operand_2;
          end;
        result := new Vector(tmp_result);
      end;
      
      static function __pow(self_vector: Vector; exponent: real): Vector;
      var 
        tmp_result: array of real;
      begin
        tmp_result := new real[self_vector.size()];
        tmp_result.Initialize();
        for var index := 0 to self_vector.size-1 do
        begin
          var operand_1 := self_vector[index];
          tmp_result[index] := operand_1 ** exponent;
        end;
        result := new Vector(tmp_result);
      end;
    
    public
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
      
      function dot(other_vector: Vector): real;
      begin
        result := 0.0;
        for var index := 0 to self.coordinates.Length-1 do
          begin
            var operand_1 := self.coordinates[index];
            var operand_2 := other_vector[index];
            result += operand_1 * operand_2;
          end;
      end;
      
      function sum(): real;
      begin
        result := 0.0;
        for var index := 0 to self.coordinates.Length-1 do
          begin
            var operand_1 := self.coordinates[index];
            result += operand_1;
          end;
      end;
      
      static function operator+(self_vector, other_vector: Vector): Vector;
      begin
        result := __add(self_vector, other_vector);
      end;  
      
      static function operator-(self_vector, other_vector: Vector): Vector;
      begin
        result := __sub(self_vector, other_vector);
      end;
      
      static function operator*(self_vector, other_vector: Vector): Vector;
      begin
        result := __mul(self_vector, other_vector);
      end;      
      
      static function operator*(self_vector: Vector; other_operand: real): Vector;
      begin
        result := __mul(self_vector, new Vector(other_operand));
      end;

      static function operator**(self_vector: Vector; exponent: real): Vector;
      begin
        result := __pow(self_vector, exponent);
      end;
         
      function back(): real;
      begin
        result := self.coordinates[self.coordinates.Length-1];
      end;
      
      procedure push_back(x: real);
      begin
        self.coordinates := self.coordinates.Append(x).ToArray();  
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
