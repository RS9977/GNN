long unsigned int __GIMPLE (ssa)
ran1_get (void * vstate)
{
  long unsigned int j;
  const long int t;
  const long int h;
  const long unsigned int x;
  struct ran1_state_t * state;
  long unsigned int D_3138;
  long int _1;
  long unsigned int _2;
  long unsigned int _3;
  long int _4;
  long int _5;
  long unsigned int _6;
  long unsigned int _7;
  long int _8;
  long unsigned int _9;
  long unsigned int _10;
  long int _11;
  long int _12;
  long unsigned int _13;
  long unsigned int _14;
  long int _15;
  long int _16;
  long unsigned int _17;
  long unsigned int _18;
  long unsigned int _19;
  long unsigned int _20;
  long unsigned int _21;
  long unsigned int _34;

  __BB(2):
  state_24 = vstate_23(D);
  x_26 = state_24->x;
  _1 = 127773l;
  _2 = (long unsigned int) _1;
  _3 = x_26 / _2;
  h_27 = (const long int) _3;
  _4 = 127773l;
  _5 = h_27 * _4;
  _6 = (long unsigned int) _5;
  _7 = x_26 - _6;
  _8 = 16807l;
  _9 = (long unsigned int) _8;
  _10 = _7 * _9;
  _11 = 2836l;
  _12 = h_27 * _11;
  _13 = (long unsigned int) _12;
  _14 = _10 - _13;
  t_28 = (const long int) _14;
  if (t_28 < 0l)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _15 = 2147483647l;
  _16 = t_28 + _15;
  _17 = (long unsigned int) _16;
  state_24->x = _17;
  goto __BB5;

  __BB(4):
  _18 = (long unsigned int) t_28;
  state_24->x = _18;
  goto __BB5;

  __BB(5):
  _19 = state_24->n;
  j_31 = _19 / 67108864ul;
  _20 = state_24->shuffle[j_31];
  state_24->n = _20;
  _21 = state_24->x;
  state_24->shuffle[j_31] = _21;
  _34 = state_24->n;
  goto __BB6;

  __BB(6):
L3:
  return _34;

}


double __GIMPLE (ssa)
ran1_get_double (void * vstate)
{
  float x;
  float x_max;
  double D_3133;
  long unsigned int _1;
  float _2;
  double _3;
  double _9;
  double _10;

  __BB(2):
  x_max_4 = 9.9999988079071044921875e-1;
  _1 = ran1_get (vstate_6(D));
  _2 = (float) _1;
  x_8 = _2 / 2.147483648e+9;
  if (x_8 > x_max_4)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _10 = (double) x_max_4;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4):
  _9 = (double) x_8;
  goto __BB5;

  __BB(5):
  _3 = __PHI (__BB3: _10, __BB4: _9);
L2:
  return _3;

}


void __GIMPLE (ssa)
ran1_set (void * vstate, long unsigned int s)
{
  long int t;
  long int h;
  long int t;
  long int h;
  int i;
  struct ran1_state_t * state;
  long int _1;
  long unsigned int _2;
  long unsigned int _3;
  long int _4;
  long int _5;
  long unsigned int _6;
  long unsigned int _7;
  long int _8;
  long unsigned int _9;
  long unsigned int _10;
  long int _11;
  long int _12;
  long unsigned int _13;
  long unsigned int _14;
  long int _15;
  long int _16;
  long unsigned int _17;
  long unsigned int _18;
  long int _19;
  long int _20;
  long unsigned int _21;
  long unsigned int _22;
  long int _23;
  long unsigned int _24;
  long unsigned int _25;
  long int _26;
  long int _27;
  long unsigned int _28;
  long unsigned int _29;
  long int _30;

  __BB(2):
  state_40 = vstate_39(D);
  if (s_41(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_42 = 1ul;
  goto __BB4;

  __BB(4):
  s_31 = __PHI (__BB2: s_41(D), __BB3: s_42);
  i_43 = 0;
  goto __BB8;

  __BB(5):
  _1 = 127773l;
  _2 = (long unsigned int) _1;
  _3 = s_32 / _2;
  h_54 = (long int) _3;
  _4 = 127773l;
  _5 = h_54 * _4;
  _6 = (long unsigned int) _5;
  _7 = s_32 - _6;
  _8 = 16807l;
  _9 = (long unsigned int) _8;
  _10 = _7 * _9;
  _11 = 2836l;
  _12 = h_54 * _11;
  _13 = (long unsigned int) _12;
  _14 = _10 - _13;
  t_55 = (long int) _14;
  if (t_55 < 0l)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  _15 = 2147483647l;
  t_56 = t_55 + _15;
  goto __BB7;

  __BB(7):
  t_36 = __PHI (__BB5: t_55, __BB6: t_56);
  s_57 = (long unsigned int) t_36;
  i_58 = i_34 + 1;
  goto __BB8;

  __BB(8,loop_header(1)):
  s_32 = __PHI (__BB4: s_31, __BB7: s_57);
  i_34 = __PHI (__BB4: i_43, __BB7: i_58);
  if (i_34 <= 7)
    goto __BB5;
  else
    goto __BB9;

  __BB(9):
  i_44 = 31;
  goto __BB13;

  __BB(10):
  _16 = 127773l;
  _17 = (long unsigned int) _16;
  _18 = s_33 / _17;
  h_48 = (long int) _18;
  _19 = 127773l;
  _20 = h_48 * _19;
  _21 = (long unsigned int) _20;
  _22 = s_33 - _21;
  _23 = 16807l;
  _24 = (long unsigned int) _23;
  _25 = _22 * _24;
  _26 = 2836l;
  _27 = h_48 * _26;
  _28 = (long unsigned int) _27;
  _29 = _25 - _28;
  t_49 = (long int) _29;
  if (t_49 < 0l)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _30 = 2147483647l;
  t_50 = t_49 + _30;
  goto __BB12;

  __BB(12):
  t_37 = __PHI (__BB10: t_49, __BB11: t_50);
  s_51 = (long unsigned int) t_37;
  state_40->shuffle[i_35] = s_51;
  i_53 = i_35 + _Literal (int) -1;
  goto __BB13;

  __BB(13,loop_header(2)):
  s_33 = __PHI (__BB9: s_32, __BB12: s_51);
  i_35 = __PHI (__BB9: i_44, __BB12: i_53);
  if (i_35 >= 0)
    goto __BB10;
  else
    goto __BB14;

  __BB(14):
  state_40->x = s_33;
  state_40->n = s_33;
  goto __BB15;

  __BB(15):
L12:
  return;

}


