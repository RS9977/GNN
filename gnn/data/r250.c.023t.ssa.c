long unsigned int __GIMPLE (ssa)
r250_get (void * vstate)
{
  int i;
  int j;
  long unsigned int k;
  struct r250_state_t * state;
  long unsigned int D_3130;
  long unsigned int _1;
  long unsigned int _2;
  int _3;
  long unsigned int _16;

  __BB(2):
  state_7 = vstate_6(D);
  i_9 = state_7->i;
  if (i_9 > 146)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  j_11 = i_9 + _Literal (int) -147;
  goto __BB5;

  __BB(4):
  j_10 = i_9 + 103;
  goto __BB5;

  __BB(5):
  j_4 = __PHI (__BB3: j_11, __BB4: j_10);
  _1 = state_7->x[i_9];
  _2 = state_7->x[j_4];
  k_12 = _1 ^ _2;
  state_7->x[i_9] = k_12;
  if (i_9 > 248)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  state_7->i = 0;
  goto __BB8;

  __BB(7):
  _3 = i_9 + 1;
  state_7->i = _3;
  goto __BB8;

  __BB(8):
  _16 = k_12;
  goto __BB9;

  __BB(9):
L6:
  return _16;

}


double __GIMPLE (ssa)
r250_get_double (void * vstate)
{
  double D_3122;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = r250_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 4.294967296e+9;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
r250_set (void * vstate, long unsigned int s)
{
  int k;
  long unsigned int mask;
  long unsigned int msb;
  int i;
  struct r250_state_t * state;
  long unsigned int _1;
  int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;

  __BB(2):
  state_16 = vstate_15(D);
  if (s_17(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_18 = 1ul;
  goto __BB4;

  __BB(4):
  s_7 = __PHI (__BB2: s_17(D), __BB3: s_18);
  state_16->i = 0;
  i_21 = 0;
  goto __BB6;

  __BB(5):
  _1 = s_8 * 69069ul;
  s_31 = _1 & 4294967295ul;
  state_16->x[i_9] = s_31;
  i_33 = i_9 + 1;
  goto __BB6;

  __BB(6,loop_header(1)):
  s_8 = __PHI (__BB4: s_7, __BB5: s_31);
  i_9 = __PHI (__BB4: i_21, __BB5: i_33);
  if (i_9 <= 249)
    goto __BB5;
  else
    goto __BB7;

  __BB(7):
  msb_22 = 2147483648ul;
  mask_23 = 4294967295ul;
  i_24 = 0;
  goto __BB9;

  __BB(8):
  _2 = i_10 * 7;
  k_25 = _2 + 3;
  _3 = state_16->x[k_25];
  _4 = mask_12 & _3;
  state_16->x[k_25] = _4;
  _5 = state_16->x[k_25];
  _6 = msb_11 | _5;
  state_16->x[k_25] = _6;
  mask_28 = mask_12 >> 1;
  msb_29 = msb_11 >> 1;
  i_30 = i_10 + 1;
  goto __BB9;

  __BB(9,loop_header(2)):
  i_10 = __PHI (__BB7: i_24, __BB8: i_30);
  msb_11 = __PHI (__BB7: msb_22, __BB8: msb_29);
  mask_12 = __PHI (__BB7: mask_23, __BB8: mask_28);
  if (i_10 <= 31)
    goto __BB8;
  else
    goto __BB10;

  __BB(10):
  goto __BB11;

  __BB(11):
L8:
  return;

}


