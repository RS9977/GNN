long unsigned int __GIMPLE (ssa)
slatec_get (void * vstate)
{
  struct slatec_state_t * state;
  long int y1;
  long int y0;
  long unsigned int D_3116;
  long int _1;
  long int _2;
  long int _3;
  long int _4;
  long int _5;
  long int _6;
  long int _7;
  long int _8;
  long int _9;
  long int _10;
  long int _11;
  long int _12;
  long int _13;
  long int _14;
  long int _15;
  long int _16;
  long int _17;
  long int _18;
  long int _19;
  long int _20;
  long int _21;
  long unsigned int _31;

  __BB(2):
  state_23 = vstate_22(D);
  _1 = state_23->x0;
  _2 = 1029l;
  y0_25 = _1 * _2;
  _3 = state_23->x1;
  _4 = 1536l;
  _5 = _3 * _4;
  _6 = state_23->x0;
  _7 = state_23->x1;
  _8 = _6 - _7;
  _9 = 507l;
  _10 = _8 * _9;
  _11 = _5 + _10;
  y1_26 = y0_25 + _11;
  _12 = 1731l;
  y0_27 = y0_25 + _12;
  _13 = y0_27 % 2048l;
  state_23->x0 = _13;
  _14 = state_23->x0;
  _15 = y0_27 - _14;
  _16 = _15 / 2048l;
  y1_29 = y1_26 + _16;
  _17 = y1_29 % 2048l;
  state_23->x1 = _17;
  _18 = state_23->x1;
  _19 = _18 * 2048l;
  _20 = state_23->x0;
  _21 = _19 + _20;
  _31 = (long unsigned int) _21;
  goto __BB3;

  __BB(3):
L0:
  return _31;

}


double __GIMPLE (ssa)
slatec_get_double (void * vstate)
{
  double D_3114;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = slatec_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 4.194304e+6;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
slatec_set (void * vstate, long unsigned int s)
{
  struct slatec_state_t * state;
  long int _1;
  long int _2;
  long unsigned int _3;
  long int _4;
  long int _5;
  long int _6;
  long unsigned int _7;
  long unsigned int _8;
  long unsigned int _9;
  long int _10;

  __BB(2):
  state_12 = vstate_11(D);
  s_14 = s_13(D) & 7ul;
  _1 = 4194304l;
  _2 = _1 / 8l;
  _3 = (long unsigned int) _2;
  s_15 = s_14 * _3;
  _4 = (long int) s_15;
  _5 = _4 & 2047l;
  state_12->x0 = _5;
  _6 = state_12->x0;
  _7 = (long unsigned int) _6;
  _8 = s_15 - _7;
  _9 = _8 / 2048ul;
  _10 = (long int) _9;
  state_12->x1 = _10;
  return;

}


