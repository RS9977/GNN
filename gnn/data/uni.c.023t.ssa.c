long unsigned int __GIMPLE (ssa)
uni_get (void * vstate)
{
  long int k;
  const int j;
  const int i;
  struct uni_state_t * state;
  long unsigned int D_3134;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  unsigned int _4;
  long int _5;
  long unsigned int _6;
  int _7;
  int _8;
  int _9;
  int _10;
  long unsigned int _26;

  __BB(2):
  state_15 = vstate_14(D);
  i_17 = state_15->i;
  j_18 = state_15->j;
  _1 = state_15->m[i_17];
  _2 = state_15->m[j_18];
  _3 = _1 - _2;
  k_19 = (long int) _3;
  if (k_19 < 0l)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _4 = 32767u;
  _5 = (long int) _4;
  k_20 = k_19 + _5;
  goto __BB4;

  __BB(4):
  k_11 = __PHI (__BB2: k_19, __BB3: k_20);
  _6 = (long unsigned int) k_11;
  state_15->m[j_18] = _6;
  if (i_17 == 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  state_15->i = 16;
  goto __BB7;

  __BB(6):
  _7 = state_15->i;
  _8 = _7 + _Literal (int) -1;
  state_15->i = _8;
  goto __BB7;

  __BB(7):
  if (j_18 == 0)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  state_15->j = 16;
  goto __BB10;

  __BB(9):
  _9 = state_15->j;
  _10 = _9 + _Literal (int) -1;
  state_15->j = _10;
  goto __BB10;

  __BB(10):
  _26 = (long unsigned int) k_11;
  goto __BB11;

  __BB(11):
L8:
  return _26;

}


double __GIMPLE (ssa)
uni_get_double (void * vstate)
{
  double D_3124;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = uni_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 3.2767e+4;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
uni_set (void * vstate, long unsigned int s)
{
  struct uni_state_t * state;
  unsigned int j1;
  unsigned int j0;
  unsigned int k1;
  unsigned int k0;
  unsigned int seed;
  unsigned int i;
  long unsigned int _1;
  unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  unsigned int _5;
  unsigned int _6;
  unsigned int _7;
  unsigned int _8;
  unsigned int _9;
  unsigned int _10;
  unsigned int _11;
  unsigned int _12;
  unsigned int _13;
  unsigned int _14;
  unsigned int _15;
  unsigned int _16;
  unsigned int _17;
  unsigned int _18;
  unsigned int _19;
  unsigned int _20;
  long unsigned int _21;

  __BB(2):
  state_27 = vstate_26(D);
  _1 = s_28(D) * 2ul;
  s_29 = _1 + 1ul;
  _2 = 32767u;
  _3 = (long unsigned int) _2;
  _4 = __MIN (s_29, _3);
  seed_30 = (unsigned int) _4;
  _5 = 256u;
  k0_31 = 9069u % _5;
  _6 = 256u;
  k1_32 = 9069u / _6;
  _7 = 256u;
  j0_33 = seed_30 % _7;
  _8 = 256u;
  j1_34 = seed_30 / _8;
  i_35 = 0u;
  goto __BB4;

  __BB(3):
  seed_39 = j0_23 * k0_31;
  _9 = 256u;
  _10 = seed_39 / _9;
  _11 = j1_24 * k0_31;
  _12 = j0_23 * k1_32;
  _13 = _11 + _12;
  _14 = _10 + _13;
  _15 = 256u;
  _16 = _15 / 2u;
  j1_40 = _14 % _16;
  _17 = 256u;
  j0_41 = seed_39 % _17;
  _18 = 256u;
  _19 = j1_40 * _18;
  _20 = j0_41 + _19;
  _21 = (long unsigned int) _20;
  state_27->m[i_22] = _21;
  i_43 = i_22 + 1u;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_22 = __PHI (__BB2: i_35, __BB3: i_43);
  j0_23 = __PHI (__BB2: j0_33, __BB3: j0_41);
  j1_24 = __PHI (__BB2: j1_34, __BB3: j1_40);
  if (i_22 <= 16u)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  state_27->i = 4;
  state_27->j = 16;
  goto __BB6;

  __BB(6):
L3:
  return;

}


