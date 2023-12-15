long unsigned int __GIMPLE (ssa)
gfsr4_get (void * vstate)
{
  struct gfsr4_state_t * state;
  long unsigned int D_3128;
  int _1;
  int _2;
  int _3;
  int _4;
  int _5;
  int _6;
  long unsigned int _7;
  int _8;
  int _9;
  int _10;
  long unsigned int _11;
  long unsigned int _12;
  int _13;
  int _14;
  int _15;
  long unsigned int _16;
  long unsigned int _17;
  int _18;
  int _19;
  int _20;
  long unsigned int _21;
  int _22;
  long unsigned int _23;
  long unsigned int _29;

  __BB(2):
  state_25 = vstate_24(D);
  _1 = state_25->nd;
  _2 = _1 + 1;
  _3 = _2 & 16383;
  state_25->nd = _3;
  _4 = state_25->nd;
  _5 = _4 + 15913;
  _6 = _5 & 16383;
  _7 = state_25->ra[_6];
  _8 = state_25->nd;
  _9 = _8 + 14798;
  _10 = _9 & 16383;
  _11 = state_25->ra[_10];
  _12 = _7 ^ _11;
  _13 = state_25->nd;
  _14 = _13 + 9396;
  _15 = _14 & 16383;
  _16 = state_25->ra[_15];
  _17 = _12 ^ _16;
  _18 = state_25->nd;
  _19 = _18 + 6695;
  _20 = _19 & 16383;
  _21 = state_25->ra[_20];
  _22 = state_25->nd;
  _23 = _17 ^ _21;
  state_25->ra[_22] = _23;
  _29 = state_25->ra[_22];
  goto __BB3;

  __BB(3):
L0:
  return _29;

}


double __GIMPLE (ssa)
gfsr4_get_double (void * vstate)
{
  double D_3126;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = gfsr4_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 4.294967296e+9;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
gfsr4_set (void * vstate, long unsigned int s)
{
  int k;
  long unsigned int bit;
  long unsigned int t;
  long unsigned int mask;
  long unsigned int msb;
  int j;
  int i;
  struct gfsr4_state_t * state;
  long unsigned int _1;
  long unsigned int _2;
  int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;

  __BB(2):
  state_22 = vstate_21(D);
  msb_23 = 2147483648ul;
  mask_24 = 4294967295ul;
  if (s_25(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_26 = 4357ul;
  goto __BB4;

  __BB(4):
  s_8 = __PHI (__BB2: s_25(D), __BB3: s_26);
  i_27 = 0;
  goto __BB11;

  __BB(5):
  t_37 = 0ul;
  bit_38 = msb_23;
  j_39 = 0;
  goto __BB9;

  __BB(6):
  _1 = s_9 * 69069ul;
  s_42 = _1 & 4294967295ul;
  _2 = s_42 & msb_23;
  if (_2 != 0ul)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  t_43 = t_17 | bit_18;
  goto __BB8;

  __BB(8):
  t_16 = __PHI (__BB6: t_17, __BB7: t_43);
  bit_44 = bit_18 >> 1;
  j_45 = j_13 + 1;
  goto __BB9;

  __BB(9,loop_header(3)):
  s_9 = __PHI (__BB5: s_10, __BB8: s_42);
  j_13 = __PHI (__BB5: j_39, __BB8: j_45);
  t_17 = __PHI (__BB5: t_37, __BB8: t_16);
  bit_18 = __PHI (__BB5: bit_38, __BB8: bit_44);
  if (j_13 <= 31)
    goto __BB6;
  else
    goto __BB10;

  __BB(10):
  state_22->ra[i_11] = t_17;
  i_41 = i_11 + 1;
  goto __BB11;

  __BB(11,loop_header(1)):
  s_10 = __PHI (__BB4: s_8, __BB10: s_9);
  i_11 = __PHI (__BB4: i_27, __BB10: i_41);
  if (i_11 <= 16383)
    goto __BB5;
  else
    goto __BB12;

  __BB(12):
  i_29 = 0;
  goto __BB14;

  __BB(13):
  _3 = i_12 * 3;
  k_31 = _3 + 7;
  _4 = state_22->ra[k_31];
  _5 = mask_15 & _4;
  state_22->ra[k_31] = _5;
  _6 = state_22->ra[k_31];
  _7 = msb_14 | _6;
  state_22->ra[k_31] = _7;
  mask_34 = mask_15 >> 1;
  msb_35 = msb_14 >> 1;
  i_36 = i_12 + 1;
  goto __BB14;

  __BB(14,loop_header(2)):
  i_12 = __PHI (__BB12: i_29, __BB13: i_36);
  msb_14 = __PHI (__BB12: msb_23, __BB13: msb_35);
  mask_15 = __PHI (__BB12: mask_24, __BB13: mask_34);
  if (i_12 <= 31)
    goto __BB13;
  else
    goto __BB15;

  __BB(15):
  state_22->nd = i_12;
  return;

}


