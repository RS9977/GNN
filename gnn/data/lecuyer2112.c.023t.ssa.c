long unsigned int __GIMPLE (ssa)
ran_get (void * vstate)
{
  long int r;
  long int y;
  struct ran_state_t * state;
  long unsigned int D_3113;
  long unsigned int _1;
  long int _2;
  long int _3;
  long int _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _17;

  __BB(2):
  state_10 = vstate_9(D);
  _1 = state_10->x;
  y_12 = (long int) _1;
  _2 = y_12 / 52774l;
  r_13 = _2 * 3791l;
  _3 = y_12 % 52774l;
  _4 = _3 * 40692l;
  y_14 = _4 - r_13;
  if (y_14 < 0l)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _5 = (long unsigned int) y_14;
  _6 = _5 + 2147483399ul;
  y_15 = (long int) _6;
  goto __BB4;

  __BB(4):
  y_8 = __PHI (__BB2: y_14, __BB3: y_15);
  _7 = (long unsigned int) y_8;
  state_10->x = _7;
  _17 = state_10->x;
  goto __BB5;

  __BB(5):
L2:
  return _17;

}


double __GIMPLE (ssa)
ran_get_double (void * vstate)
{
  struct ran_state_t * state;
  double D_3109;
  long unsigned int _1;
  double _2;
  double _7;

  __BB(2):
  state_4 = vstate_3(D);
  _1 = ran_get (state_4);
  _2 = (double) _1;
  _7 = _2 / 2.147483399e+9;
  goto __BB3;

  __BB(3):
L0:
  return _7;

}


void __GIMPLE (ssa)
ran_set (void * vstate, long unsigned int s)
{
  struct ran_state_t * state;
  long unsigned int _1;
  long unsigned int _2;

  __BB(2):
  state_5 = vstate_4(D);
  _1 = s_6(D) % 2147483399ul;
  if (_1 == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_7 = 1ul;
  goto __BB4;

  __BB(4):
  s_3 = __PHI (__BB2: s_6(D), __BB3: s_7);
  _2 = s_3 % 2147483399ul;
  state_5->x = _2;
  goto __BB5;

  __BB(5):
L2:
  return;

}


