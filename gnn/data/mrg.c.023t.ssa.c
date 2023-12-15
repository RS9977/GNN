long unsigned int __GIMPLE (ssa)
mrg_get (void * vstate)
{
  long int h5;
  long int p5;
  long int h1;
  long int p1;
  struct mrg_state_t * state;
  long unsigned int D_3129;
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
  long unsigned int _51;

  __BB(2):
  state_37 = vstate_36(D);
  _1 = state_37->x5;
  _2 = 20554l;
  h5_39 = _1 / _2;
  _3 = state_37->x5;
  _4 = 20554l;
  _5 = h5_39 * _4;
  _6 = _3 - _5;
  _7 = 104480l;
  _8 = _6 * _7;
  _9 = 1727l;
  _10 = h5_39 * _9;
  p5_40 = _8 - _10;
  if (p5_40 > 0l)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _11 = 2147483647l;
  p5_41 = p5_40 - _11;
  goto __BB4;

  __BB(4):
  p5_34 = __PHI (__BB2: p5_40, __BB3: p5_41);
  _12 = state_37->x1;
  _13 = 20l;
  h1_42 = _12 / _13;
  _14 = state_37->x1;
  _15 = 20l;
  _16 = h1_42 * _15;
  _17 = _14 - _16;
  _18 = 107374182l;
  _19 = _17 * _18;
  _20 = 7l;
  _21 = h1_42 * _20;
  p1_43 = _19 - _21;
  if (p1_43 < 0l)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _22 = 2147483647l;
  p1_44 = p1_43 + _22;
  goto __BB6;

  __BB(6):
  p1_33 = __PHI (__BB4: p1_43, __BB5: p1_44);
  _23 = state_37->x4;
  state_37->x5 = _23;
  _24 = state_37->x3;
  state_37->x4 = _24;
  _25 = state_37->x2;
  state_37->x3 = _25;
  _26 = state_37->x1;
  state_37->x2 = _26;
  _27 = p1_33 + p5_34;
  state_37->x1 = _27;
  _28 = state_37->x1;
  if (_28 < 0l)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _29 = state_37->x1;
  _30 = 2147483647l;
  _31 = _29 + _30;
  state_37->x1 = _31;
  goto __BB8;

  __BB(8):
  _32 = state_37->x1;
  _51 = (long unsigned int) _32;
  goto __BB9;

  __BB(9):
L6:
  return _51;

}


double __GIMPLE (ssa)
mrg_get_double (void * vstate)
{
  double D_3121;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = mrg_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 2.147483647e+9;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
mrg_set (void * vstate, long unsigned int s)
{
  struct mrg_state_t * state;
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

  __BB(2):
  state_28 = vstate_27(D);
  if (s_29(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_30 = 1ul;
  goto __BB4;

  __BB(4):
  s_26 = __PHI (__BB2: s_29(D), __BB3: s_30);
  _1 = s_26 * 69069ul;
  s_31 = _1 & 4294967295ul;
  _2 = 2147483647l;
  _3 = (long unsigned int) _2;
  _4 = s_31 % _3;
  _5 = (long int) _4;
  state_28->x1 = _5;
  _6 = s_31 * 69069ul;
  s_34 = _6 & 4294967295ul;
  _7 = 2147483647l;
  _8 = (long unsigned int) _7;
  _9 = s_34 % _8;
  _10 = (long int) _9;
  state_28->x2 = _10;
  _11 = s_34 * 69069ul;
  s_36 = _11 & 4294967295ul;
  _12 = 2147483647l;
  _13 = (long unsigned int) _12;
  _14 = s_36 % _13;
  _15 = (long int) _14;
  state_28->x3 = _15;
  _16 = s_36 * 69069ul;
  s_38 = _16 & 4294967295ul;
  _17 = 2147483647l;
  _18 = (long unsigned int) _17;
  _19 = s_38 % _18;
  _20 = (long int) _19;
  state_28->x4 = _20;
  _21 = s_38 * 69069ul;
  s_40 = _21 & 4294967295ul;
  _22 = 2147483647l;
  _23 = (long unsigned int) _22;
  _24 = s_40 % _23;
  _25 = (long int) _24;
  state_28->x5 = _25;
  mrg_get (state_28);
  mrg_get (state_28);
  mrg_get (state_28);
  mrg_get (state_28);
  mrg_get (state_28);
  mrg_get (state_28);
  goto __BB5;

  __BB(5):
L2:
  return;

}


