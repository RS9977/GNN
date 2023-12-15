long unsigned int __GIMPLE (ssa)
transputer_get (void * vstate)
{
  struct transputer_state_t * state;
  long unsigned int D_3108;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _8;

  __BB(2):
  state_5 = vstate_4(D);
  _1 = state_5->x;
  _2 = _1 * 1664525ul;
  _3 = _2 & 4294967295ul;
  state_5->x = _3;
  _8 = state_5->x;
  goto __BB3;

  __BB(3):
L0:
  return _8;

}


double __GIMPLE (ssa)
transputer_get_double (void * vstate)
{
  double D_3106;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = transputer_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 4.294967296e+9;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
transputer_set (void * vstate, long unsigned int s)
{
  struct transputer_state_t * state;

  __BB(2):
  state_3 = vstate_2(D);
  if (s_4(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_5 = 1ul;
  goto __BB4;

  __BB(4):
  s_1 = __PHI (__BB2: s_4(D), __BB3: s_5);
  state_3->x = s_1;
  goto __BB5;

  __BB(5):
L2:
  return;

}


