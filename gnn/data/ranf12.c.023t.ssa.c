void __GIMPLE (ssa)
ranf_advance (void * vstate)
{
  long unsigned int r;
  const long unsigned int x2;
  const long unsigned int x1;
  const long unsigned int x0;
  struct ranf_state_t * state;
  short unsigned int _1;
  short unsigned int _2;
  short unsigned int _3;
  short unsigned int _4;
  long unsigned int _5;
  short unsigned int _6;
  short unsigned int _7;
  long unsigned int _8;
  long unsigned int _9;
  short unsigned int _10;
  long unsigned int _11;
  long unsigned int _12;
  long unsigned int _13;
  short unsigned int _14;
  short unsigned int _15;
  long unsigned int _16;
  long unsigned int _17;
  short unsigned int _18;
  long unsigned int _19;
  long unsigned int _20;
  long unsigned int _21;
  short unsigned int _22;
  long unsigned int _23;
  long unsigned int _24;
  long unsigned int _25;
  short unsigned int _26;

  __BB(2):
  state_28 = vstate_27(D);
  _1 = state_28->x0;
  x0_30 = (const long unsigned int) _1;
  _2 = state_28->x1;
  x1_31 = (const long unsigned int) _2;
  _3 = state_28->x2;
  x2_32 = (const long unsigned int) _3;
  _4 = _Literal (const short unsigned int) 45429;
  _5 = (long unsigned int) _4;
  r_33 = x0_30 * _5;
  _6 = (short unsigned int) r_33;
  state_28->x0 = _6;
  r_35 = r_33 >> 16;
  _7 = _Literal (const short unsigned int) 45429;
  _8 = (long unsigned int) _7;
  _9 = x1_31 * _8;
  _10 = _Literal (const short unsigned int) 41703;
  _11 = (long unsigned int) _10;
  _12 = x0_30 * _11;
  _13 = _9 + _12;
  r_36 = r_35 + _13;
  _14 = (short unsigned int) r_36;
  state_28->x1 = _14;
  r_38 = r_36 >> 16;
  _15 = _Literal (const short unsigned int) 45429;
  _16 = (long unsigned int) _15;
  _17 = x2_32 * _16;
  _18 = _Literal (const short unsigned int) 41703;
  _19 = (long unsigned int) _18;
  _20 = x1_31 * _19;
  _21 = _17 + _20;
  _22 = _Literal (const short unsigned int) 10357;
  _23 = (long unsigned int) _22;
  _24 = x0_30 * _23;
  _25 = _21 + _24;
  r_39 = r_38 + _25;
  _26 = (short unsigned int) r_39;
  state_28->x2 = _26;
  return;

}


long unsigned int __GIMPLE (ssa)
ranf_get (void * vstate)
{
  struct ranf_state_t * state;
  long unsigned int x2;
  long unsigned int x1;
  long unsigned int D_4232;
  short unsigned int _1;
  short unsigned int _2;
  long unsigned int _3;
  long unsigned int _10;

  __BB(2):
  state_5 = vstate_4(D);
  ranf_advance (state_5);
  _1 = state_5->x1;
  x1_8 = (long unsigned int) _1;
  _2 = state_5->x2;
  x2_9 = (long unsigned int) _2;
  _3 = x2_9 << 16;
  _10 = x1_8 + _3;
  goto __BB3;

  __BB(3):
L0:
  return _10;

}


void __GIMPLE (ssa)
ranf_set (void * vstate, long unsigned int s)
{
  long unsigned int b2;
  long unsigned int b1;
  long unsigned int b0;
  long unsigned int r;
  short unsigned int x2;
  short unsigned int x1;
  short unsigned int x0;
  struct ranf_state_t * state;
  short unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  short unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  long unsigned int _9;
  short unsigned int _10;
  long unsigned int _11;
  long unsigned int _12;
  long unsigned int _13;
  long unsigned int _14;
  long unsigned int _15;
  long unsigned int _16;
  long unsigned int _17;
  long unsigned int _18;
  short unsigned int _19;

  __BB(2):
  state_24 = vstate_23(D);
  b0_25 = 55005ul;
  b1_26 = 47252ul;
  b2_27 = 23790ul;
  if (s_28(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  x0_32 = _Literal (short unsigned int) 40145;
  x1_33 = _Literal (short unsigned int) 21500;
  x2_34 = _Literal (short unsigned int) 38018;
  goto __BB5;

  __BB(4):
  _1 = (short unsigned int) s_28(D);
  x0_29 = _1 | _Literal (short unsigned int) 1;
  _2 = s_28(D) >> 16;
  x1_30 = (short unsigned int) _2;
  x2_31 = _Literal (short unsigned int) 0;
  goto __BB5;

  __BB(5):
  x0_20 = __PHI (__BB3: x0_32, __BB4: x0_29);
  x1_21 = __PHI (__BB3: x1_33, __BB4: x1_30);
  x2_22 = __PHI (__BB3: x2_34, __BB4: x2_31);
  _3 = (long unsigned int) x0_20;
  r_35 = b0_25 * _3;
  _4 = (short unsigned int) r_35;
  state_24->x0 = _4;
  r_38 = r_35 >> 16;
  _5 = (long unsigned int) x1_21;
  _6 = b0_25 * _5;
  _7 = (long unsigned int) x0_20;
  _8 = b1_26 * _7;
  _9 = _6 + _8;
  r_39 = r_38 + _9;
  _10 = (short unsigned int) r_39;
  state_24->x1 = _10;
  r_41 = r_39 >> 16;
  _11 = (long unsigned int) x2_22;
  _12 = b0_25 * _11;
  _13 = (long unsigned int) x1_21;
  _14 = b1_26 * _13;
  _15 = _12 + _14;
  _16 = (long unsigned int) x0_20;
  _17 = b2_27 * _16;
  _18 = _15 + _17;
  r_42 = r_41 + _18;
  _19 = (short unsigned int) r_42;
  state_24->x2 = _19;
  goto __BB6;

  __BB(6):
L3:
  return;

}


double __GIMPLE (ssa)
ranf_get_double (void * vstate)
{
  struct ranf_state_t * state;
  double D_4234;
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
  ranf_advance (state_12);
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


