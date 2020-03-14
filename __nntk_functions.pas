unit __nntk_functions;
uses vector_math;
 
  function relu(const input: Vector): Vector;
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
  
  function relu_derivative(const input: Vector): Vector;
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

end.