uses nntk, vector_math;

const
  EPOCHS_COUNT = 100;
  LEARNING_RATE = 0.15;
  DROPOUT_PROBABILITY = 0.0;
  BATCH_SIZE = 1;
  SEED = 0;

var
  INITIALIZING_WEIGHTS_RANGE: System.Tuple<real, real> := (-10.0, 10.0);
  ACTIVATION_FUNCTIONS: array of nntk.functions_type := 
  (
    nntk.functions.tanh,
    nntk.functions.tanh
  );
  ACTIVATION_FUNCTIONS_DERIVATIVES: array of nntk.functions_type := 
  (
    nntk.functions.tanh_derivative, 
    nntk.functions.tanh_derivative
  );


begin
  var ipt := new List<vector_math.Vector>;
  ipt.add(new Vector(0, 0));
  ipt.Add(new Vector(0, 1));
  ipt.Add(new Vector(1, 0));
  ipt.Add(new Vector(1, 1));
  var opt := new List<vector>;
  opt.Add(new Vector(0));
  opt.Add(new Vector(1)); 
  opt.Add(new Vector(1));
  opt.Add(new Vector(0));

  var size := new Vector(2, 4, 1); 
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
           
  println('Neural Network output for (1, 0): ', nn.run(new Vector(1, 0)));  
end.