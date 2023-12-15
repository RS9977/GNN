void __GIMPLE (ssa)
rand48_advance (void * vstate)
{
  long unsigned int a;
  const long unsigned int x2;
  const long unsigned int x1;
  const long unsigned int x0;
  struct rand48_state_t * state;
  short unsigned int _1;
  short unsigned int _2;
  short unsigned int _3;
  short unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  short unsigned int _7;
  long unsigned int _8;
  short unsigned int _9;
  short unsigned int _10;
  long unsigned int _11;
  long unsigned int _12;
  short unsigned int _13;
  long unsigned int _14;
  long unsigned int _15;
  long unsigned int _16;
  short unsigned int _17;
  short unsigned int _18;
  long unsigned int _19;
  long unsigned int _20;
  short unsigned int _21;
  long unsigned int _22;
  long unsigned int _23;
  long unsigned int _24;
  short unsigned int _25;
  long unsigned int _26;
  long unsigned int _27;
  long unsigned int _28;
  short unsigned int _29;

  __BB(2):
  state_31 = vstate_30(D);
  _1 = state_31->x0;
  x0_33 = (const long unsigned int) _1;
  _2 = state_31->x1;
  x1_34 = (const long unsigned int) _2;
  _3 = state_31->x2;
  x2_35 = (const long unsigned int) _3;
  _4 = _Literal (const short unsigned int) 58989;
  _5 = (long unsigned int) _4;
  _6 = x0_33 * _5;
  _7 = _Literal (const short unsigned int) 11;
  _8 = (long unsigned int) _7;
  a_36 = _6 + _8;
  _9 = (short unsigned int) a_36;
  state_31->x0 = _9;
  a_38 = a_36 >> 16;
  _10 = _Literal (const short unsigned int) 58989;
  _11 = (long unsigned int) _10;
  _12 = x1_34 * _11;
  _13 = _Literal (const short unsigned int) 57068;
  _14 = (long unsigned int) _13;
  _15 = x0_33 * _14;
  _16 = _12 + _15;
  a_39 = a_38 + _16;
  _17 = (short unsigned int) a_39;
  state_31->x1 = _17;
  a_41 = a_39 >> 16;
  _18 = _Literal (const short unsigned int) 58989;
  _19 = (long unsigned int) _18;
  _20 = x2_35 * _19;
  _21 = _Literal (const short unsigned int) 57068;
  _22 = (long unsigned int) _21;
  _23 = x1_34 * _22;
  _24 = _20 + _23;
  _25 = _Literal (const short unsigned int) 5;
  _26 = (long unsigned int) _25;
  _27 = x0_33 * _26;
  _28 = _24 + _27;
  a_42 = a_41 + _28;
  _29 = (short unsigned int) a_42;
  state_31->x2 = _29;
  return;

}


long unsigned int __GIMPLE (ssa)
rand48_get (void * vstate)
{
  struct rand48_state_t * state;
  long unsigned int x2;
  long unsigned int x1;
  long unsigned int D_4226;
  short unsigned int _1;
  short unsigned int _2;
  long unsigned int _3;
  long unsigned int _10;

  __BB(2):
  state_5 = vstate_4(D);
  rand48_advance (state_5);
  _1 = state_5->x2;
  x2_8 = (long unsigned int) _1;
  _2 = state_5->x1;
  x1_9 = (long unsigned int) _2;
  _3 = x2_8 << 16;
  _10 = x1_9 + _3;
  goto __BB3;

  __BB(3):
L0:
  return _10;

}


void __GIMPLE (ssa)
rand48_set (void * vstate, long unsigned int s)
{
  struct rand48_state_t * state;
  short unsigned int _1;
  long unsigned int _2;
  short unsigned int _3;

  __BB(2):
  state_6 = vstate_5(D);
  if (s_7(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  state_6->x0 = _Literal (short unsigned int) 13070;
  state_6->x1 = _Literal (short unsigned int) 43981;
  state_6->x2 = _Literal (short unsigned int) 4660;
  goto __BB5;

  __BB(4):
  state_6->x0 = _Literal (short unsigned int) 13070;
  _1 = (short unsigned int) s_7(D);
  state_6->x1 = _1;
  _2 = s_7(D) >> 16;
  _3 = (short unsigned int) _2;
  state_6->x2 = _3;
  goto __BB5;

  __BB(5):
  goto __BB6;

  __BB(6):
L3:
  return;

}


double __GIMPLE (ssa)
rand48_get_double (void * vstate)
{
  struct rand48_state_t * state;
  double D_4228;
  short unsigned int _1;
  double _2;
  double _3;
  short unsigned int _4;
  double _5;
  double _6;
  double _7;
  short unsigned int _8;
  double _9;
  double _10;
  double _18;

  __BB(2):
  state_12 = vstate_11(D);
  rand48_advance (state_12);
  _1 = state_12->x2;
  _2 = (double) _1;
  _3 = ldexp (_2, _Literal (int) -16);
  _4 = state_12->x1;
  _5 = (double) _4;
  _6 = ldexp (_5, _Literal (int) -32);
  _7 = _3 + _6;
  _8 = state_12->x0;
  _9 = (double) _8;
  _10 = ldexp (_9, _Literal (int) -48);
  _18 = _7 + _10;
  goto __BB3;

  __BB(3):
L0:
  return _18;

}


