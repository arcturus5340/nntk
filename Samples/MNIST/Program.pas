uses nntk, vector_math;

const
  LEARNING_RATE = 0.005;
  EPOCHS_COUNT = 300;
  DROPOUT_PROBABILITY = 0.5;

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
  var nn := new Neural_Network(size);
  nn.train(ipt, opt, 
           EPOCHS_COUNT, 
           LEARNING_RATE, 
           nntk.Functions.relu, 
           nntk.Functions.relu_derivative, 
           DROPOUT_PROBABILITY);
  
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