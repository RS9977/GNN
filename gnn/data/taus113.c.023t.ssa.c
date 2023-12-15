long unsigned int __GIMPLE (ssa)
taus113_get (void * vstate)
{
  long unsigned int b4;
  long unsigned int b3;
  long unsigned int b2;
  long unsigned int b1;
  struct taus113_state_t * state;
  long unsigned int D_3115;
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
  long unsigned int _30;
  long unsigned int _31;
  long unsigned int _32;
  long unsigned int _33;
  long unsigned int _34;
  long unsigned int _35;
  long unsigned int _36;
  long unsigned int _37;
  long unsigned int _38;
  long unsigned int _39;
  long unsigned int _40;
  long unsigned int _41;
  long unsigned int _42;
  long unsigned int _54;

  __BB(2):
  state_44 = vstate_43(D);
  _1 = state_44->z1;
  _2 = _1 << 6u;
  _3 = _2 & 4294967295ul;
  _4 = state_44->z1;
  _5 = _3 ^ _4;
  b1_46 = _5 >> 13u;
  _6 = state_44->z1;
  _7 = _6 << 18u;
  _8 = _7 & 4294443008ul;
  _9 = b1_46 ^ _8;
  state_44->z1 = _9;
  _10 = state_44->z2;
  _11 = _10 << 2u;
  _12 = _11 & 4294967295ul;
  _13 = state_44->z2;
  _14 = _12 ^ _13;
  b2_48 = _14 >> 27u;
  _15 = state_44->z2;
  _16 = _15 << 2u;
  _17 = _16 & 4294967264ul;
  _18 = b2_48 ^ _17;
  state_44->z2 = _18;
  _19 = state_44->z3;
  _20 = _19 << 13u;
  _21 = _20 & 4294967295ul;
  _22 = state_44->z3;
  _23 = _21 ^ _22;
  b3_50 = _23 >> 21u;
  _24 = state_44->z3;
  _25 = _24 << 7u;
  _26 = _25 & 4294965248ul;
  _27 = b3_50 ^ _26;
  state_44->z3 = _27;
  _28 = state_44->z4;
  _29 = _28 << 3u;
  _30 = _29 & 4294967295ul;
  _31 = state_44->z4;
  _32 = _30 ^ _31;
  b4_52 = _32 >> 12u;
  _33 = state_44->z4;
  _34 = _33 << 13u;
  _35 = _34 & 4293918720ul;
  _36 = b4_52 ^ _35;
  state_44->z4 = _36;
  _37 = state_44->z1;
  _38 = state_44->z2;
  _39 = _37 ^ _38;
  _40 = state_44->z3;
  _41 = _39 ^ _40;
  _42 = state_44->z4;
  _54 = _41 ^ _42;
  goto __BB3;

  __BB(3):
L0:
  return _54;

}


double __GIMPLE (ssa)
taus113_get_double (void * vstate)
{
  double D_3113;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = taus113_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 4.294967296e+9;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
taus113_set (void * vstate, long unsigned int s)
{
  struct taus113_state_t * state;
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

  __BB(2):
  state_30 = vstate_29(D);
  if (s_31(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_32 = 1ul;
  goto __BB4;

  __BB(4):
  s_24 = __PHI (__BB2: s_31(D), __BB3: s_32);
  _1 = s_24 * 69069ul;
  _2 = _1 & 4294967295ul;
  state_30->z1 = _2;
  _3 = state_30->z1;
  if (_3 <= 1ul)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _4 = state_30->z1;
  _5 = _4 + 2ul;
  state_30->z1 = _5;
  goto __BB6;

  __BB(6):
  _6 = state_30->z1;
  _7 = _6 * 69069ul;
  _8 = _7 & 4294967295ul;
  state_30->z2 = _8;
  _9 = state_30->z2;
  if (_9 <= 7ul)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _10 = state_30->z2;
  _11 = _10 + 8ul;
  state_30->z2 = _11;
  goto __BB8;

  __BB(8):
  _12 = state_30->z2;
  _13 = _12 * 69069ul;
  _14 = _13 & 4294967295ul;
  state_30->z3 = _14;
  _15 = state_30->z3;
  if (_15 <= 15ul)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  _16 = state_30->z3;
  _17 = _16 + 16ul;
  state_30->z3 = _17;
  goto __BB10;

  __BB(10):
  _18 = state_30->z3;
  _19 = _18 * 69069ul;
  _20 = _19 & 4294967295ul;
  state_30->z4 = _20;
  _21 = state_30->z4;
  if (_21 <= 127ul)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _22 = state_30->z4;
  _23 = _22 + 128ul;
  state_30->z4 = _23;
  goto __BB12;

  __BB(12):
  taus113_get (state_30);
  taus113_get (state_30);
  taus113_get (state_30);
  taus113_get (state_30);
  taus113_get (state_30);
  taus113_get (state_30);
  taus113_get (state_30);
  taus113_get (state_30);
  taus113_get (state_30);
  taus113_get (state_30);
  goto __BB13;

  __BB(13):
L10:
  return;

}


