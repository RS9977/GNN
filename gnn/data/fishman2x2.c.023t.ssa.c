long unsigned int __GIMPLE (ssa)
ran_get (void * vstate)
{
  long int r;
  long int y;
  struct ran_state_t * state;
  long unsigned int D_3121;
  long unsigned int iftmp.6;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
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
  long unsigned int _18;
  long unsigned int _19;
  long unsigned int _20;
  long unsigned int _21;
  long unsigned int _22;
  long unsigned int _23;
  long unsigned int _24;
  long unsigned int _25;
  long unsigned int _26;
  long unsigned int _27;
  long unsigned int _28;
  long unsigned int _29;
  long unsigned int _32;
  long unsigned int _44;
  long unsigned int _45;
  long unsigned int _47;

  __BB(2):
  state_34 = vstate_33(D);
  _1 = state_34->x;
  _2 = _1 / 44488ul;
  _3 = _2 * 3399ul;
  r_36 = (long int) _3;
  _4 = state_34->x;
  _5 = _4 % 44488ul;
  _6 = _5 * 48271ul;
  _7 = (long unsigned int) r_36;
  _8 = _6 - _7;
  y_37 = (long int) _8;
  if (y_37 < 0l)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _9 = (long unsigned int) y_37;
  _10 = _9 + 2147483647ul;
  y_38 = (long int) _10;
  goto __BB4;

  __BB(4):
  y_30 = __PHI (__BB2: y_37, __BB3: y_38);
  _11 = (long unsigned int) y_30;
  state_34->x = _11;
  _12 = state_34->y;
  _13 = _12 / 52774ul;
  _14 = _13 * 3791ul;
  r_40 = (long int) _14;
  _15 = state_34->y;
  _16 = _15 % 52774ul;
  _17 = _16 * 40692ul;
  _18 = (long unsigned int) r_40;
  _19 = _17 - _18;
  y_41 = (long int) _19;
  if (y_41 < 0l)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _20 = (long unsigned int) y_41;
  _21 = _20 + 2147483399ul;
  y_42 = (long int) _21;
  goto __BB6;

  __BB(6):
  y_31 = __PHI (__BB4: y_41, __BB5: y_42);
  _22 = (long unsigned int) y_31;
  state_34->y = _22;
  _23 = state_34->x;
  _24 = state_34->y;
  if (_23 > _24)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _25 = state_34->x;
  _26 = state_34->y;
  _45 = _25 - _26;
  goto __BB9;

  __BB(8):
  _27 = state_34->x;
  _28 = state_34->y;
  _29 = _27 - _28;
  _44 = _29 + 2147483647ul;
  goto __BB9;

  __BB(9):
  _32 = __PHI (__BB7: _45, __BB8: _44);
  state_34->z = _32;
  _47 = state_34->z;
  goto __BB10;

  __BB(10):
L7:
  return _47;

}


double __GIMPLE (ssa)
ran_get_double (void * vstate)
{
  struct ran_state_t * state;
  double D_3111;
  long unsigned int _1;
  double _2;
  double _7;

  __BB(2):
  state_4 = vstate_3(D);
  _1 = ran_get (state_4);
  _2 = (double) _1;
  _7 = _2 / 2.147483647e+9;
  goto __BB3;

  __BB(3):
L0:
  return _7;

}


void __GIMPLE (ssa)
ran_set (void * vstate, long unsigned int s)
{
  struct ran_state_t * state;
  long unsigned int iftmp.7;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  long unsigned int _9;
  long unsigned int _10;
  long unsigned int _11;
  long unsigned int _13;
  long unsigned int _21;
  long unsigned int _22;

  __BB(2):
  state_15 = vstate_14(D);
  _1 = s_16(D) % 2147483647ul;
  if (_1 == 0ul)
    goto __BB4;
  else
    goto __BB3;

  __BB(3):
  _2 = s_16(D) % 2147483399ul;
  if (_2 == 0ul)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  s_17 = 1ul;
  goto __BB5;

  __BB(5):
  s_12 = __PHI (__BB3: s_16(D), __BB4: s_17);
  _3 = s_12 % 2147483647ul;
  state_15->x = _3;
  _4 = s_12 % 2147483399ul;
  state_15->y = _4;
  _5 = state_15->x;
  _6 = state_15->y;
  if (_5 > _6)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  _7 = state_15->x;
  _8 = state_15->y;
  _22 = _7 - _8;
  goto __BB8;

  __BB(7):
  _9 = state_15->x;
  _10 = state_15->y;
  _11 = _9 - _10;
  _21 = _11 + 2147483647ul;
  goto __BB8;

  __BB(8):
  _13 = __PHI (__BB6: _22, __BB7: _21);
  state_15->z = _13;
  goto __BB9;

  __BB(9):
L6:
  return;

}


