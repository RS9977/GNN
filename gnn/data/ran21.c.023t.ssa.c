long unsigned int __GIMPLE (ssa)
ran2_get (void * vstate)
{
  long int delta;
  long unsigned int j;
  long int t2;
  long int h2;
  long int t1;
  long int h1;
  const long unsigned int y;
  const long unsigned int x;
  struct ran2_state_t * state;
  long unsigned int D_3150;
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
  long unsigned int _16;
  long unsigned int _17;
  long int _18;
  long int _19;
  long unsigned int _20;
  long unsigned int _21;
  long int _22;
  long unsigned int _23;
  long unsigned int _24;
  long int _25;
  long int _26;
  long unsigned int _27;
  long unsigned int _28;
  long int _29;
  long int _30;
  long unsigned int _31;
  long unsigned int _32;
  long unsigned int _33;
  long unsigned int _34;
  long unsigned int _35;
  long unsigned int _36;
  long int _37;
  long int _38;
  long unsigned int _39;
  long unsigned int _40;
  long unsigned int _62;

  __BB(2):
  state_45 = vstate_44(D);
  x_47 = state_45->x;
  y_48 = state_45->y;
  _1 = 53668l;
  _2 = (long unsigned int) _1;
  _3 = x_47 / _2;
  h1_49 = (long int) _3;
  _4 = 53668l;
  _5 = h1_49 * _4;
  _6 = (long unsigned int) _5;
  _7 = x_47 - _6;
  _8 = 40014l;
  _9 = (long unsigned int) _8;
  _10 = _7 * _9;
  _11 = 12211l;
  _12 = h1_49 * _11;
  _13 = (long unsigned int) _12;
  _14 = _10 - _13;
  t1_50 = (long int) _14;
  _15 = 52774l;
  _16 = (long unsigned int) _15;
  _17 = y_48 / _16;
  h2_51 = (long int) _17;
  _18 = 52774l;
  _19 = h2_51 * _18;
  _20 = (long unsigned int) _19;
  _21 = y_48 - _20;
  _22 = 40692l;
  _23 = (long unsigned int) _22;
  _24 = _21 * _23;
  _25 = 3791l;
  _26 = h2_51 * _25;
  _27 = (long unsigned int) _26;
  _28 = _24 - _27;
  t2_52 = (long int) _28;
  if (t1_50 < 0l)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _29 = 2147483563l;
  t1_53 = t1_50 + _29;
  goto __BB4;

  __BB(4):
  t1_41 = __PHI (__BB2: t1_50, __BB3: t1_53);
  if (t2_52 < 0l)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _30 = 2147483399l;
  t2_54 = t2_52 + _30;
  goto __BB6;

  __BB(6):
  t2_42 = __PHI (__BB4: t2_52, __BB5: t2_54);
  _31 = (long unsigned int) t1_41;
  state_45->x = _31;
  _32 = (long unsigned int) t2_42;
  state_45->y = _32;
  _33 = state_45->n;
  j_57 = _33 / 67108862ul;
  _34 = state_45->shuffle[j_57];
  _35 = (long unsigned int) t2_42;
  _36 = _34 - _35;
  delta_58 = (long int) _36;
  if (delta_58 <= 0l)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _37 = 2147483563l;
  _38 = _37 + _Literal (long int) -1l;
  delta_59 = delta_58 + _38;
  goto __BB8;

  __BB(8):
  delta_43 = __PHI (__BB6: delta_58, __BB7: delta_59);
  _39 = (long unsigned int) delta_43;
  state_45->n = _39;
  _40 = (long unsigned int) t1_41;
  state_45->shuffle[j_57] = _40;
  _62 = state_45->n;
  goto __BB9;

  __BB(9):
L6:
  return _62;

}


double __GIMPLE (ssa)
ran2_get_double (void * vstate)
{
  float x;
  float x_max;
  double D_3142;
  long unsigned int _1;
  float _2;
  double _3;
  double _9;
  double _10;

  __BB(2):
  x_max_4 = 9.9999988079071044921875e-1;
  _1 = ran2_get (vstate_6(D));
  _2 = (float) _1;
  x_8 = _2 / 2.14748352e+9;
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
ran2_set (void * vstate, long unsigned int s)
{
  long int t;
  long int h;
  long int t;
  long int h;
  int i;
  struct ran2_state_t * state;
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
  state_40->y = s_31;
  i_45 = 0;
  goto __BB8;

  __BB(5):
  _1 = 53668l;
  _2 = (long unsigned int) _1;
  _3 = s_32 / _2;
  h_55 = (long int) _3;
  _4 = 53668l;
  _5 = h_55 * _4;
  _6 = (long unsigned int) _5;
  _7 = s_32 - _6;
  _8 = 40014l;
  _9 = (long unsigned int) _8;
  _10 = _7 * _9;
  _11 = 12211l;
  _12 = h_55 * _11;
  _13 = (long unsigned int) _12;
  _14 = _10 - _13;
  t_56 = (long int) _14;
  if (t_56 < 0l)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  _15 = 2147483563l;
  t_57 = t_56 + _15;
  goto __BB7;

  __BB(7):
  t_36 = __PHI (__BB5: t_56, __BB6: t_57);
  s_58 = (long unsigned int) t_36;
  i_59 = i_34 + 1;
  goto __BB8;

  __BB(8,loop_header(1)):
  s_32 = __PHI (__BB4: s_31, __BB7: s_58);
  i_34 = __PHI (__BB4: i_45, __BB7: i_59);
  if (i_34 <= 7)
    goto __BB5;
  else
    goto __BB9;

  __BB(9):
  i_46 = 31;
  goto __BB13;

  __BB(10):
  _16 = 53668l;
  _17 = (long unsigned int) _16;
  _18 = s_33 / _17;
  h_49 = (long int) _18;
  _19 = 53668l;
  _20 = h_49 * _19;
  _21 = (long unsigned int) _20;
  _22 = s_33 - _21;
  _23 = 40014l;
  _24 = (long unsigned int) _23;
  _25 = _22 * _24;
  _26 = 12211l;
  _27 = h_49 * _26;
  _28 = (long unsigned int) _27;
  _29 = _25 - _28;
  t_50 = (long int) _29;
  if (t_50 < 0l)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _30 = 2147483563l;
  t_51 = t_50 + _30;
  goto __BB12;

  __BB(12):
  t_37 = __PHI (__BB10: t_50, __BB11: t_51);
  s_52 = (long unsigned int) t_37;
  state_40->shuffle[i_35] = s_52;
  i_54 = i_35 + _Literal (int) -1;
  goto __BB13;

  __BB(13,loop_header(2)):
  s_33 = __PHI (__BB9: s_32, __BB12: s_52);
  i_35 = __PHI (__BB9: i_46, __BB12: i_54);
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


