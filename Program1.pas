uses nntk, vector_math;

begin
  var size := new Vector<integer>(3, 4, 1); 
  var nn := new Neural_Network(size);
  var ipt := new Vector<Vector<real>>(new Vector<real>(1, 0, 1), 
                                      new Vector<real>(0, 1, 1), 
                                      new Vector<real>(0, 0, 1),
                                      new Vector<real>(1, 1, 1));
  var opt := new vector<vector<real>>(new Vector<real>(0), 
                                      new Vector<real>(1), 
                                      new Vector<real>(0),
                                      new Vector<real>(1));
  nn.learn(ipt, opt, 1000);
  print(nn.run(new Vector<real>(0, 0, 0)));
end.