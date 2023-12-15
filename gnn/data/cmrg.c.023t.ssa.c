long unsigned int __GIMPLE (ssa)
cmrg_get (void * vstate)
{
  long int p1;
  long int h1;
  long int p3;
  long int h3;
  long int p2;
  long int h2;
  long int p3;
  long int h3;
  struct cmrg_state_t * state;
  long unsigned int D_3149;
  long int _1;
  long int _2;
  long int _3;
  long int _4;
  long int _5;
  long int _6;
  long int _7;
  long int _8;
  long int _9;
  long int _10;
  long int _11;
  long int _12;
  long int _13;
  long int _14;
  long int _15;
  long int _16;
  long int _17;
  long int _18;
  long int _19;
  long int _20;
  long int _21;
  long int _22;
  long int _23;
  long int _24;
  long int _25;
  long int _26;
  long int _27;
  long int _28;
  long int _29;
  long int _30;
  long int _31;
  long int _32;
  long int _33;
  long int _34;
  long int _35;
  long int _36;
  long int _37;
  long int _38;
  long int _39;
  long int _40;
  long int _41;
  long int _42;
  long int _43;
  long int _44;
  long int _45;
  long int _46;
  long int _47;
  long int _48;
  long int _49;
  long int _50;
  long int _51;
  long int _52;
  long int _53;
  long int _54;
  long int _55;
  long int _56;
  long int _57;
  long int _58;
  long int _59;
  long int _60;
  long int _61;
  long int _62;
  long int _63;
  long int _64;
  long int _65;
  long int _66;
  long int _67;
  long int _68;
  long int _69;
  long int _70;
  long unsigned int _75;
  long unsigned int _101;
  long unsigned int _102;

  __BB(2):
  state_79 = vstate_78(D);
  _1 = state_79->x3;
  _2 = 11714l;
  h3_81 = _1 / _2;
  _3 = _Literal (const long int) -183326l;
  _4 = -_3;
  _5 = state_79->x3;
  _6 = 11714l;
  _7 = h3_81 * _6;
  _8 = _5 - _7;
  _9 = _4 * _8;
  _10 = 2883l;
  _11 = h3_81 * _10;
  p3_82 = _9 - _11;
  _12 = state_79->x2;
  _13 = 33921l;
  h2_83 = _12 / _13;
  _14 = state_79->x2;
  _15 = 33921l;
  _16 = h2_83 * _15;
  _17 = _14 - _16;
  _18 = 63308l;
  _19 = _17 * _18;
  _20 = 12979l;
  _21 = h2_83 * _20;
  p2_84 = _19 - _21;
  if (p3_82 < 0l)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _22 = 2147483647l;
  p3_85 = p3_82 + _22;
  goto __BB4;

  __BB(4):
  p3_71 = __PHI (__BB2: p3_82, __BB3: p3_85);
  if (p2_84 < 0l)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _23 = 2147483647l;
  p2_86 = p2_84 + _23;
  goto __BB6;

  __BB(6):
  p2_72 = __PHI (__BB4: p2_84, __BB5: p2_86);
  _24 = state_79->x2;
  state_79->x3 = _24;
  _25 = state_79->x1;
  state_79->x2 = _25;
  _26 = p2_72 - p3_71;
  state_79->x1 = _26;
  _27 = state_79->x1;
  if (_27 < 0l)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _28 = state_79->x1;
  _29 = 2147483647l;
  _30 = _28 + _29;
  state_79->x1 = _30;
  goto __BB8;

  __BB(8):
  _31 = state_79->y3;
  _32 = 3976l;
  h3_91 = _31 / _32;
  _33 = _Literal (const long int) -539608l;
  _34 = -_33;
  _35 = state_79->y3;
  _36 = 3976l;
  _37 = h3_91 * _36;
  _38 = _35 - _37;
  _39 = _34 * _38;
  _40 = 2071l;
  _41 = h3_91 * _40;
  p3_92 = _39 - _41;
  _42 = state_79->y1;
  _43 = 24919l;
  h1_93 = _42 / _43;
  _44 = state_79->y1;
  _45 = 24919l;
  _46 = h1_93 * _45;
  _47 = _44 - _46;
  _48 = 86098l;
  _49 = _47 * _48;
  _50 = 7417l;
  _51 = h1_93 * _50;
  p1_94 = _49 - _51;
  if (p3_92 < 0l)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  _52 = 2145483479l;
  p3_95 = p3_92 + _52;
  goto __BB10;

  __BB(10):
  p3_73 = __PHI (__BB8: p3_92, __BB9: p3_95);
  if (p1_94 < 0l)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _53 = 2145483479l;
  p1_96 = p1_94 + _53;
  goto __BB12;

  __BB(12):
  p1_74 = __PHI (__BB10: p1_94, __BB11: p1_96);
  _54 = state_79->y2;
  state_79->y3 = _54;
  _55 = state_79->y1;
  state_79->y2 = _55;
  _56 = p1_74 - p3_73;
  state_79->y1 = _56;
  _57 = state_79->y1;
  if (_57 < 0l)
    goto __BB13;
  else
    goto __BB14;

  __BB(13):
  _58 = state_79->y1;
  _59 = 2145483479l;
  _60 = _58 + _59;
  state_79->y1 = _60;
  goto __BB14;

  __BB(14):
  _61 = state_79->x1;
  _62 = state_79->y1;
  if (_61 < _62)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  _63 = state_79->x1;
  _64 = state_79->y1;
  _65 = _63 - _64;
  _66 = 2147483647l;
  _67 = _65 + _66;
  _102 = (long unsigned int) _67;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(16):
  _68 = state_79->x1;
  _69 = state_79->y1;
  _70 = _68 - _69;
  _101 = (long unsigned int) _70;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(17):
  _75 = __PHI (__BB15: _102, __BB16: _101);
L14:
  return _75;

}


double __GIMPLE (ssa)
cmrg_get_double (void * vstate)
{
  double D_3133;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = cmrg_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 2.147483647e+9;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
cmrg_set (void * vstate, long unsigned int s)
{
  struct cmrg_state_t * state;
  long unsigned int _1;
  long int _2;
  long unsigned int _3;
  long unsigned int _4;
  long int _5;
  long unsigned int _6;
  long int _7;
  long unsigned int _8;
  long unsigned int _9;
  long int _10;
  long unsigned int _11;
  long int _12;
  long unsigned int _13;
  long unsigned int _14;
  long int _15;
  long unsigned int _16;
  long int _17;
  long unsigned int _18;
  long unsigned int _19;
  long int _20;
  long unsigned int _21;
  long int _22;
  long unsigned int _23;
  long unsigned int _24;
  long int _25;
  long unsigned int _26;
  long int _27;
  long unsigned int _28;
  long unsigned int _29;
  long int _30;

  __BB(2):
  state_33 = vstate_32(D);
  if (s_34(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_35 = 1ul;
  goto __BB4;

  __BB(4):
  s_31 = __PHI (__BB2: s_34(D), __BB3: s_35);
  _1 = s_31 * 69069ul;
  s_36 = _1 & 4294967295ul;
  _2 = 2147483647l;
  _3 = (long unsigned int) _2;
  _4 = s_36 % _3;
  _5 = (long int) _4;
  state_33->x1 = _5;
  _6 = s_36 * 69069ul;
  s_39 = _6 & 4294967295ul;
  _7 = 2147483647l;
  _8 = (long unsigned int) _7;
  _9 = s_39 % _8;
  _10 = (long int) _9;
  state_33->x2 = _10;
  _11 = s_39 * 69069ul;
  s_41 = _11 & 4294967295ul;
  _12 = 2147483647l;
  _13 = (long unsigned int) _12;
  _14 = s_41 % _13;
  _15 = (long int) _14;
  state_33->x3 = _15;
  _16 = s_41 * 69069ul;
  s_43 = _16 & 4294967295ul;
  _17 = 2145483479l;
  _18 = (long unsigned int) _17;
  _19 = s_43 % _18;
  _20 = (long int) _19;
  state_33->y1 = _20;
  _21 = s_43 * 69069ul;
  s_45 = _21 & 4294967295ul;
  _22 = 2145483479l;
  _23 = (long unsigned int) _22;
  _24 = s_45 % _23;
  _25 = (long int) _24;
  state_33->y2 = _25;
  _26 = s_45 * 69069ul;
  s_47 = _26 & 4294967295ul;
  _27 = 2145483479l;
  _28 = (long unsigned int) _27;
  _29 = s_47 % _28;
  _30 = (long int) _29;
  state_33->y3 = _30;
  cmrg_get (state_33);
  cmrg_get (state_33);
  cmrg_get (state_33);
  cmrg_get (state_33);
  cmrg_get (state_33);
  cmrg_get (state_33);
  cmrg_get (state_33);
  return;

}


