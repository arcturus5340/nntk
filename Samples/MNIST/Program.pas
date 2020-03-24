uses nntk, vector_math;

const
  LEARNING_RATE = 0.0015;
  EPOCHS_COUNT = 300;
  DROPOUT_PROBABILITY = 0.5;
  BATCH_SIZE = 8;
  SEED = 0;
  
var
  INITIALIZING_WEIGHTS_RANGE: System.Tuple<real, real> := (-1.0, 1.0);
  ACTIVATION_FUNCTIONS: array of nntk.functions_type := 
  (nntk.functions.tanh, nntk.functions.softmax);
  ACTIVATION_FUNCTIONS_DERIVATIVES: array of nntk.functions_type := 
  (nntk.functions.tanh_derivative, nntk.functions.softmax_derivative);
  LOSS_FUNCTION := nntk.loss_functions.mse;
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
  
  var test_ipt := new List<Vector>;
  foreach var str in ReadallLines('test_images.txt') do
  begin  
    var x := new Vector;
    foreach var num in str.Split(' ') do
      x.push_back(real.Parse(num));
    test_ipt.Add(x);
  end;
  var test_opt := new List<Vector>;
  foreach var str in ReadallLines('test_labels.txt') do
  begin  
    var x := new Vector;
    foreach var num in str.Split(' ') do
      x.push_back(real.Parse(num));
    test_opt.Add(x);
  end; 

  var size := new Vector(784, 100, 10); 
  var nn := new Neural_Network(size, 
                               INITIALIZING_WEIGHTS_RANGE, 
                               ACTIVATION_FUNCTIONS, 
                               ACTIVATION_FUNCTIONS_DERIVATIVES,
                               SEED);
  nn.train(ipt, opt, test_ipt, test_opt,
           EPOCHS_COUNT, 
           LEARNING_RATE, 
           LOSS_FUNCTION,
           DROPOUT_PROBABILITY,
           BATCH_SIZE);
           
end.