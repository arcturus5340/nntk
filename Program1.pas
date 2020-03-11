 uses nntk, vector_math;

begin
  var ipt := new List<Vector>;
  foreach var str in ReadallLines('images.txt') do
  begin  
    var x := new Vector;
    foreach var num in str.Split(' ') do
      x.push_back(real.Parse(num));
    ipt.Add(x);
  end; 
  var opt := new list<Vector>;
  foreach var str in ReadallLines('labels.txt') do
  begin  
    var x := new Vector;
    foreach var num in str.Split(' ') do
      x.push_back(real.Parse(num));
    opt.Add(x);
  end; 
  var size := new Vector(784, 100, 10); 
  var d := Milliseconds;
  var nn := new Neural_Network(size);
  writeln(Milliseconds-d);
//  var ipt := new List<Vector>;
//  ipt.add(new Vector(0, 0));
//  ipt.Add(new Vector(0, 1));
//  ipt.Add(new Vector(1, 0));
//  ipt.Add(new Vector(1, 1));
//  var opt := new List<vector>;
//  opt.Add(new Vector(0));
//  opt.Add(new Vector(1)); 
//  opt.Add(new Vector(1));
//  opt.Add(new Vector(0));
  var e := Milliseconds;
  nn.train(ipt, opt, 100);
  writeln(Milliseconds-e);
  
//  print(nn.run(new Vector(0, 0)));
  
  
  var my_model := nn.get_model();
  ipt := new List<Vector>;
  foreach var str in ReadallLines('test_images.txt') do
  begin  
    var x := new Vector;
    foreach var num in str.Split(' ') do
      x.push_back(real.Parse(num));
    ipt.Add(x);
  end;
  opt := new List<Vector>;
  foreach var str in ReadallLines('test_labels.txt') do
  begin  
    var x := new Vector;
    foreach var num in str.Split(' ') do
      x.push_back(real.Parse(num));
    opt.Add(x);
  end; 
  var error := 0.0;
  println(ipt.Count);
  for var index := 0 to ipt.Count-1 do 
    error += ((my_model(ipt[index]) - opt[index]) ** 2).sum();
  println(error / ipt.Count);
end.