uses nntk, vector_math;

const
  LEARNING_RATE = 0.15;
  EPOCHS_COUNT = 500;
  DROPOUT_PROBABILITY = 0.0;

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
  var nn := new Neural_Network(size);
  nn.train(ipt, opt, 
           EPOCHS_COUNT, 
           LEARNING_RATE, 
           nntk.Functions.tanh, 
           nntk.Functions.tanh_derivative, 
           DROPOUT_PROBABILITY);
  
  println('Neural Network output for (0, 1): ', nn.run(new Vector(0, 1)));  
end.