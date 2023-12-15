long unsigned int __GIMPLE (ssa)
uni32_get (void * vstate)
{
  long int k;
  const long int j;
  const long int i;
  struct uni32_state_t * state;
  long unsigned int D_3134;
  int _1;
  int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  long unsigned int _9;
  int _10;
  int _11;
  int _12;
  int _13;
  long unsigned int _29;

  __BB(2):
  state_18 = vstate_17(D);
  _1 = state_18->i;
  i_20 = (const long int) _1;
  _2 = state_18->j;
  j_21 = (const long int) _2;
  _3 = state_18->m[i_20];
  _4 = state_18->m[j_21];
  _5 = _3 - _4;
  k_22 = (long int) _5;
  if (k_22 < 0l)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _6 = (long unsigned int) k_22;
  _7 = 2147483647ul;
  _8 = _6 + _7;
  k_23 = (long int) _8;
  goto __BB4;

  __BB(4):
  k_14 = __PHI (__BB2: k_22, __BB3: k_23);
  _9 = (long unsigned int) k_14;
  state_18->m[j_21] = _9;
  if (i_20 == 0l)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  state_18->i = 16;
  goto __BB7;

  __BB(6):
  _10 = state_18->i;
  _11 = _10 + _Literal (int) -1;
  state_18->i = _11;
  goto __BB7;

  __BB(7):
  if (j_21 == 0l)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  state_18->j = 16;
  goto __BB10;

  __BB(9):
  _12 = state_18->j;
  _13 = _12 + _Literal (int) -1;
  state_18->j = _13;
  goto __BB10;

  __BB(10):
  _29 = (long unsigned int) k_14;
  goto __BB11;

  __BB(11):
L8:
  return _29;

}


double __GIMPLE (ssa)
uni32_get_double (void * vstate)
{
  double D_3124;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = uni32_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 2.147483647e+9;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
uni32_set (void * vstate, long unsigned int s)
{
  struct uni32_state_t * state;
  int i;
  long int j1;
  long int j0;
  long int k1;
  long int k0;
  long int seed;
  long int iftmp.4;
  long unsigned int _1;
  long unsigned int _2;
  unsigned long _3;
  unsigned long _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  long unsigned int _9;
  long unsigned int _10;
  long unsigned int _11;
  long unsigned int _12;
  long unsigned int _13;
  long unsigned int _14;
  long unsigned int _15;
  long unsigned int _16;
  long unsigned int _17;
  long int _18;
  long unsigned int _19;
  long int _20;
  long unsigned int _21;
  long unsigned int _22;
  long unsigned int _23;
  long unsigned int _24;
  long unsigned int _25;
  long unsigned int _26;
  long unsigned int _27;
  long unsigned int _28;
  long unsigned int _29;
  long unsigned int _30;
  long unsigned int _31;
  long unsigned int _32;
  long unsigned int _33;
  long unsigned int _34;
  long int _38;
  long int _44;
  long int _45;

  __BB(2):
  state_41 = vstate_40(D);
  _1 = 2147483647ul;
  _2 = __MIN (s_42(D), _1);
  seed_43 = (long int) _2;
  _3 = (unsigned long) seed_43;
  _4 = _3 & 1ul;
  if (_4 == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _45 = 1l;
  goto __BB5;

  __BB(4):
  _44 = 0l;
  goto __BB5;

  __BB(5):
  _38 = __PHI (__BB3: _45, __BB4: _44);
  seed_46 = seed_43 - _38;
  _5 = 65536ul;
  _6 = 9069ul % _5;
  k0_47 = (long int) _6;
  _7 = 65536ul;
  _8 = 9069ul / _7;
  k1_48 = (long int) _8;
  _9 = (long unsigned int) seed_46;
  _10 = 65536ul;
  _11 = _9 % _10;
  j0_49 = (long int) _11;
  _12 = (long unsigned int) seed_46;
  _13 = 65536ul;
  _14 = _12 / _13;
  j1_50 = (long int) _14;
  i_51 = 0;
  goto __BB7;

  __BB(6):
  seed_55 = j0_35 * k0_47;
  _15 = (long unsigned int) seed_55;
  _16 = 65536ul;
  _17 = _15 / _16;
  _18 = j1_36 * k0_47;
  _19 = (long unsigned int) _18;
  _20 = j0_35 * k1_48;
  _21 = (long unsigned int) _20;
  _22 = _19 + _21;
  _23 = _17 + _22;
  _24 = 65536ul;
  _25 = _24 / 2ul;
  _26 = _23 % _25;
  j1_56 = (long int) _26;
  _27 = (long unsigned int) seed_55;
  _28 = 65536ul;
  _29 = _27 % _28;
  j0_57 = (long int) _29;
  _30 = 65536ul;
  _31 = (long unsigned int) j1_56;
  _32 = _30 * _31;
  _33 = (long unsigned int) j0_57;
  _34 = _32 + _33;
  state_41->m[i_37] = _34;
  i_59 = i_37 + 1;
  goto __BB7;

  __BB(7,loop_header(1)):
  j0_35 = __PHI (__BB5: j0_49, __BB6: j0_57);
  j1_36 = __PHI (__BB5: j1_50, __BB6: j1_56);
  i_37 = __PHI (__BB5: i_51, __BB6: i_59);
  if (i_37 <= 16)
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  state_41->i = 4;
  state_41->j = 16;
  goto __BB9;

  __BB(9):
L6:
  return;

}


