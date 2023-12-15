long unsigned int __GIMPLE (ssa)
randu_get (void * vstate)
{
  struct randu_state_t * state;
  long unsigned int D_3109;
  long unsigned int _1;
  long int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _10;

  __BB(2):
  state_7 = vstate_6(D);
  _1 = state_7->x;
  _2 = 65539l;
  _3 = (long unsigned int) _2;
  _4 = _1 * _3;
  _5 = _4 & 2147483647ul;
  state_7->x = _5;
  _10 = state_7->x;
  goto __BB3;

  __BB(3):
L0:
  return _10;

}


double __GIMPLE (ssa)
randu_get_double (void * vstate)
{
  double D_3107;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = randu_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 2.147483648e+9;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
randu_set (void * vstate, long unsigned int s)
{
  struct randu_state_t * state;

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


