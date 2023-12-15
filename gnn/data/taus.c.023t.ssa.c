long unsigned int __GIMPLE (ssa)
taus_get (void * vstate)
{
  struct taus_state_t * state;
  long unsigned int D_3117;
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
  long unsigned int _41;

  __BB(2):
  state_36 = vstate_35(D);
  _1 = state_36->s1;
  _2 = _1 << 12;
  _3 = _2 & 4294959104ul;
  _4 = state_36->s1;
  _5 = _4 << 13;
  _6 = _5 & 4294967295ul;
  _7 = state_36->s1;
  _8 = _6 ^ _7;
  _9 = _8 >> 19;
  _10 = _3 ^ _9;
  state_36->s1 = _10;
  _11 = state_36->s2;
  _12 = _11 << 4;
  _13 = _12 & 4294967168ul;
  _14 = state_36->s2;
  _15 = _14 << 2;
  _16 = _15 & 4294967295ul;
  _17 = state_36->s2;
  _18 = _16 ^ _17;
  _19 = _18 >> 25;
  _20 = _13 ^ _19;
  state_36->s2 = _20;
  _21 = state_36->s3;
  _22 = _21 << 17;
  _23 = _22 & 4292870144ul;
  _24 = state_36->s3;
  _25 = _24 << 3;
  _26 = _25 & 4294967295ul;
  _27 = state_36->s3;
  _28 = _26 ^ _27;
  _29 = _28 >> 11;
  _30 = _23 ^ _29;
  state_36->s3 = _30;
  _31 = state_36->s1;
  _32 = state_36->s2;
  _33 = _31 ^ _32;
  _34 = state_36->s3;
  _41 = _33 ^ _34;
  goto __BB3;

  __BB(3):
L0:
  return _41;

}


double __GIMPLE (ssa)
taus_get_double (void * vstate)
{
  double D_3115;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = taus_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 4.294967296e+9;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
taus_set (void * vstate, long unsigned int s)
{
  struct taus_state_t * state;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;

  __BB(2):
  state_11 = vstate_10(D);
  if (s_12(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_13 = 1ul;
  goto __BB4;

  __BB(4):
  s_9 = __PHI (__BB2: s_12(D), __BB3: s_13);
  _1 = s_9 * 69069ul;
  _2 = _1 & 4294967295ul;
  state_11->s1 = _2;
  _3 = state_11->s1;
  _4 = _3 * 69069ul;
  _5 = _4 & 4294967295ul;
  state_11->s2 = _5;
  _6 = state_11->s2;
  _7 = _6 * 69069ul;
  _8 = _7 & 4294967295ul;
  state_11->s3 = _8;
  taus_get (state_11);
  taus_get (state_11);
  taus_get (state_11);
  taus_get (state_11);
  taus_get (state_11);
  taus_get (state_11);
  goto __BB5;

  __BB(5):
L2:
  return;

}


void __GIMPLE (ssa)
taus2_set (void * vstate, long unsigned int s)
{
  struct taus_state_t * state;
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

  __BB(2):
  state_23 = vstate_22(D);
  if (s_24(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_25 = 1ul;
  goto __BB4;

  __BB(4):
  s_18 = __PHI (__BB2: s_24(D), __BB3: s_25);
  _1 = s_18 * 69069ul;
  _2 = _1 & 4294967295ul;
  state_23->s1 = _2;
  _3 = state_23->s1;
  if (_3 <= 1ul)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _4 = state_23->s1;
  _5 = _4 + 2ul;
  state_23->s1 = _5;
  goto __BB6;

  __BB(6):
  _6 = state_23->s1;
  _7 = _6 * 69069ul;
  _8 = _7 & 4294967295ul;
  state_23->s2 = _8;
  _9 = state_23->s2;
  if (_9 <= 7ul)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _10 = state_23->s2;
  _11 = _10 + 8ul;
  state_23->s2 = _11;
  goto __BB8;

  __BB(8):
  _12 = state_23->s2;
  _13 = _12 * 69069ul;
  _14 = _13 & 4294967295ul;
  state_23->s3 = _14;
  _15 = state_23->s3;
  if (_15 <= 15ul)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  _16 = state_23->s3;
  _17 = _16 + 16ul;
  state_23->s3 = _17;
  goto __BB10;

  __BB(10):
  taus_get (state_23);
  taus_get (state_23);
  taus_get (state_23);
  taus_get (state_23);
  taus_get (state_23);
  taus_get (state_23);
  goto __BB11;

  __BB(11):
L8:
  return;

}


