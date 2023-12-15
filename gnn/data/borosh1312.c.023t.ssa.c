long unsigned int __GIMPLE (ssa)
ran_get (void * vstate)
{
  struct ran_state_t * state;
  long unsigned int D_3109;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _8;

  __BB(2):
  state_5 = vstate_4(D);
  _1 = state_5->x;
  _2 = _1 * 1812433253ul;
  _3 = _2 & 4294967295ul;
  state_5->x = _3;
  _8 = state_5->x;
  goto __BB3;

  __BB(3):
L0:
  return _8;

}


double __GIMPLE (ssa)
ran_get_double (void * vstate)
{
  struct ran_state_t * state;
  double D_3107;
  long unsigned int _1;
  double _2;
  double _7;

  __BB(2):
  state_4 = vstate_3(D);
  _1 = ran_get (state_4);
  _2 = (double) _1;
  _7 = _2 / 4.294967296e+9;
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

  __BB(2):
  state_4 = vstate_3(D);
  if (s_5(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_6 = 1ul;
  goto __BB4;

  __BB(4):
  s_2 = __PHI (__BB2: s_5(D), __BB3: s_6);
  _1 = s_2 & 4294967295ul;
  state_4->x = _1;
  goto __BB5;

  __BB(5):
L2:
  return;

}


