uses nntk, vector_math;

var
  my_model: function(input: Vector<real>): Vector<real>;

begin
  var size := new Vector<integer>(2, 2, 1); 
  var nn := new Neural_Network(size);
  var ipt := new Vector<Vector<real>>(new Vector<real>(0, 0), 
                                      new Vector<real>(0, 1), 
                                      new Vector<real>(1, 0),
                                      new Vector<real>(1, 1));
  var opt := new vector<vector<real>>(new Vector<real>(0), 
                                      new Vector<real>(1), 
                                      new Vector<real>(1),
                                      new Vector<real>(0));
  nn.learn(ipt, opt, 1000);
  print(nn.run(new Vector<real>(0, 0)));
  my_model := nn.get_model();
  print(my_model(new Vector<real>(0, 1)));
end.