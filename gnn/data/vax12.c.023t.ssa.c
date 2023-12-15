long unsigned int __GIMPLE (ssa)
vax_get (void * vstate)
{
  struct vax_state_t * state;
  long unsigned int D_3108;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _9;

  __BB(2):
  state_6 = vstate_5(D);
  _1 = state_6->x;
  _2 = _1 * 69069ul;
  _3 = _2 + 1ul;
  _4 = _3 & 4294967295ul;
  state_6->x = _4;
  _9 = state_6->x;
  goto __BB3;

  __BB(3):
L0:
  return _9;

}


double __GIMPLE (ssa)
vax_get_double (void * vstate)
{
  double D_3106;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = vax_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 4.294967296e+9;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
vax_set (void * vstate, long unsigned int s)
{
  struct vax_state_t * state;

  __BB(2):
  state_2 = vstate_1(D);
  state_2->x = s_4(D);
  goto __BB3;

  __BB(3):
L0:
  return;

}


