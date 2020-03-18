uses nntk, vector_math;

const
  LEARNING_RATE = 0.0015;
  EPOCHS_COUNT = 200;
  DROPOUT_PROBABILITY = 0.5;
  BATCH_SIZE = 100;
  SEED = 0;
  
var
  INITIALIZING_WEIGHTS_RANGE: System.Tuple<real, real> := (-1.0, 1.0);
  ACTIVATION_FUNCTIONS: array of nntk.functions_type := 
  (nntk.functions.tanh, nntk.functions.softmax);
  ACTIVATION_FUNCTIONS_DERIVATIVES: array of nntk.functions_type := 
  (nntk.functions.tanh_derivative, nntk.functions.softmax_derivative);

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
  var nn := new Neural_Network(size, 
                               INITIALIZING_WEIGHTS_RANGE, 
                               ACTIVATION_FUNCTIONS, 
                               ACTIVATION_FUNCTIONS_DERIVATIVES,
                               SEED);
  nn.train(ipt, opt, 
           EPOCHS_COUNT, 
           LEARNING_RATE, 
           DROPOUT_PROBABILITY,
           BATCH_SIZE);
  
//  var my_model := nn.get_model();
//  ipt := new List<Vector>;
//  foreach var str in ReadallLines('test_images.txt') do
//  begin  
//    var x := new Vector;
//    foreach var num in str.Split(' ') do
//      x.push_back(real.Parse(num));
//    ipt.Add(x);
//  end;
//  opt := new List<Vector>;
//  foreach var str in ReadallLines('test_labels.txt') do
//  begin  
//    var x := new Vector;
//    foreach var num in str.Split(' ') do
//      x.push_back(real.Parse(num));
//    opt.Add(x);
//  end; 
//  var error := 0.0;
//  println(ipt.Count);
//  for var index := 0 to ipt.Count-1 do 
//    println('Error: ', ((my_model(ipt[index]) - opt[index]) ** 2).sum());
end.