long unsigned int __GIMPLE (ssa)
ran0_get (void * vstate)
{
  const long int t;
  const long int h;
  const long unsigned int x;
  struct ran0_state_t * state;
  long unsigned int D_3121;
  long int _1;
  long unsigned int _2;
  long unsigned int _3;
  long int _4;
  long int _5;
  long unsigned int _6;
  long unsigned int _7;
  long int _8;
  long unsigned int _9;
  long unsigned int _10;
  long int _11;
  long int _12;
  long unsigned int _13;
  long unsigned int _14;
  long int _15;
  long int _16;
  long unsigned int _17;
  long unsigned int _18;
  long unsigned int _28;

  __BB(2):
  state_21 = vstate_20(D);
  x_23 = state_21->x;
  _1 = 127773l;
  _2 = (long unsigned int) _1;
  _3 = x_23 / _2;
  h_24 = (const long int) _3;
  _4 = 127773l;
  _5 = h_24 * _4;
  _6 = (long unsigned int) _5;
  _7 = x_23 - _6;
  _8 = 16807l;
  _9 = (long unsigned int) _8;
  _10 = _7 * _9;
  _11 = 2836l;
  _12 = h_24 * _11;
  _13 = (long unsigned int) _12;
  _14 = _10 - _13;
  t_25 = (const long int) _14;
  if (t_25 < 0l)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _15 = 2147483647l;
  _16 = t_25 + _15;
  _17 = (long unsigned int) _16;
  state_21->x = _17;
  goto __BB5;

  __BB(4):
  _18 = (long unsigned int) t_25;
  state_21->x = _18;
  goto __BB5;

  __BB(5):
  _28 = state_21->x;
  goto __BB6;

  __BB(6):
L3:
  return _28;

}


double __GIMPLE (ssa)
ran0_get_double (void * vstate)
{
  double D_3116;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = ran0_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 2.147483647e+9;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
ran0_set (void * vstate, long unsigned int s)
{
  struct ran0_state_t * state;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;

  __BB(2):
  state_6 = vstate_5(D);
  _1 = 123459876ul;
  if (s_7(D) == _1)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("ran0 should not use seed == mask", ".//ran0.c", 82, 4);
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4):
  _2 = 123459876ul;
  _3 = s_7(D) ^ _2;
  state_6->x = _3;
  goto __BB5;

  __BB(5):
L2:
  return;

}


