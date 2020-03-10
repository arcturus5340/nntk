uses nntk, vector_math;

begin
//  var ipt := new Vector<Vector>;
//  foreach var str in ReadallLines('images.txt') do
//  begin  
//    var x := new Vector;
//    foreach var num in str.Split(' ') do
//      x.push_back(real.Parse(num));
//    ipt.push_back(x);
//  end; 
//  var opt := new Vector<Vector>;
//  foreach var str in ReadallLines('labels.txt') do
//  begin  
//    var x := new Vector;
//    foreach var num in str.Split(' ') do
//      x.push_back(real.Parse(num));
//    opt.push_back(x);
//  end; 
  
  var size := new Vector(2, 4, 1); 
  var nn := new Neural_Network(size);
  var ipt := new List<Vector>;
  ipt.add(new Vector(0, 0));
  ipt.Add(new Vector(0, 1));
  ipt.Add(new Vector(1, 0));
  ipt.Add(new Vector(1, 1));
  var opt := new List<vector>;
  opt.Add(new Vector(0));
  opt.Add(new Vector(1)); 
  opt.Add(new Vector(1));
  opt.Add(new Vector(0));
  nn.learn(ipt, opt, 1000);
  print(nn.run(new Vector(0, 0)));
  
//    var o1 := new Vector(1, 2, 3, 4, 5);
//    for var i := 0 to 10000 do
//      o1.push_back(random);
//    var o2 := new Vector(5, 4, 3, 2, 1);
//    for var i := 0 to 10000 do
//      o2.push_back(random);
//    var d := Milliseconds;
//    print(o1 * o2);
//    writeln('Непараллельное перемножение матриц: ',Milliseconds-d,' миллисекунд');
     
  
//  my_model := nn.get_model();
//  ipt := new Vector<Vector>;
//  foreach var str in ReadallLines('test_images.txt') do
//  begin  
//    var x := new Vector;
//    foreach var num in str.Split(' ') do
//      x.push_back(real.Parse(num));
//    ipt.push_back(x);
//  end;
//  opt := new Vector<Vector>;
//  foreach var str in ReadallLines('test_labels.txt') do
//  begin  
//    var x := new Vector;
//    foreach var num in str.Split(' ') do
//      x.push_back(real.Parse(num));
//    opt.push_back(x);
//  end; 
//  var error := 0.0;
//  println(ipt.size);
//  for var index := 0 to ipt.size-1 do 
//    error += ((my_model(ipt[index]) - opt[index]) ** 2).sum();
//  println(error / ipt.size);
end.