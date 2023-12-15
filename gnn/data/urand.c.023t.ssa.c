double __GIMPLE (ssa)
urand ()
{
  static long unsigned int x = 1ul;
  double D_1949;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  double _6;
  double _9;

  __BB(2):
  _1 = x;
  _2 = _1 * 1103515245ul;
  _3 = _2 + 12345ul;
  _4 = _3 & 2147483647ul;
  x = _4;
  _5 = x;
  _6 = (double) _5;
  _9 = _6 / 2.147483648e+9;
  goto __BB3;

  __BB(3):
L0:
  return _9;

}


