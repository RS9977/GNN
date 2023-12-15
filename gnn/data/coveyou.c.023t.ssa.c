long unsigned int __GIMPLE (ssa)
ran_get (void * vstate)
{
  struct ran_state_t * state;
  long unsigned int D_3110;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _10;

  __BB(2):
  state_7 = vstate_6(D);
  _1 = state_7->x;
  _2 = state_7->x;
  _3 = _2 + 1ul;
  _4 = _1 * _3;
  _5 = _4 & 4294967295ul;
  state_7->x = _5;
  _10 = state_7->x;
  goto __BB3;

  __BB(3):
L0:
  return _10;

}


double __GIMPLE (ssa)
ran_get_double (void * vstate)
{
  struct ran_state_t * state;
  double D_3108;
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
  long unsigned int diff;
  struct ran_state_t * state;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;

  __BB(2):
  state_8 = vstate_7(D);
  _1 = s_9(D) & 3ul;
  _2 = _1 + 18446744073709551614ul;
  diff_10 = _2 % 4294967295ul;
  if (diff_10 != 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _3 = s_9(D) - diff_10;
  _4 = _3 & 4294967295ul;
  state_8->x = _4;
  goto __BB5;

  __BB(4):
  _5 = s_9(D) & 4294967295ul;
  state_8->x = _5;
  goto __BB5;

  __BB(5):
  goto __BB6;

  __BB(6):
L3:
  return;

}


