long unsigned int __GIMPLE (ssa)
mt_get (void * vstate)
{
  long unsigned int y;
  long unsigned int y;
  long unsigned int y;
  int kk;
  long unsigned int * const mt;
  long unsigned int k;
  struct mt_state_t * state;
  long unsigned int D_3170;
  long unsigned int iftmp.8;
  long unsigned int iftmp.5;
  long unsigned int iftmp.2;
  int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int * _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  sizetype _8;
  sizetype _9;
  sizetype _10;
  long unsigned int * _11;
  long unsigned int _12;
  long unsigned int _13;
  long unsigned int _14;
  sizetype _15;
  sizetype _16;
  sizetype _17;
  long unsigned int * _18;
  long unsigned int _19;
  long unsigned int _20;
  long unsigned int _21;
  long unsigned int _22;
  long unsigned int _23;
  long unsigned int _24;
  long unsigned int * _25;
  long unsigned int _26;
  long unsigned int _27;
  long unsigned int _28;
  long unsigned int * _29;
  long unsigned int _30;
  long unsigned int _31;
  long unsigned int _32;
  sizetype _33;
  sizetype _34;
  sizetype _35;
  long unsigned int * _36;
  long unsigned int _37;
  long unsigned int _38;
  long unsigned int _39;
  long unsigned int _40;
  long unsigned int _41;
  sizetype _42;
  long unsigned int * _43;
  long unsigned int _44;
  long unsigned int _45;
  long unsigned int _46;
  long unsigned int _47;
  long unsigned int _48;
  long unsigned int _49;
  long unsigned int * _50;
  long unsigned int _51;
  long unsigned int * _52;
  long unsigned int _53;
  long unsigned int _54;
  long unsigned int _55;
  long unsigned int _56;
  long unsigned int _57;
  long unsigned int _58;
  long unsigned int * _59;
  long unsigned int _60;
  long unsigned int _61;
  long unsigned int _62;
  long unsigned int _63;
  long unsigned int * _64;
  long unsigned int _65;
  int _66;
  long unsigned int _67;
  long unsigned int _68;
  long unsigned int * _69;
  long unsigned int _70;
  long unsigned int _71;
  long unsigned int _72;
  long unsigned int _73;
  long unsigned int _74;
  long unsigned int _75;
  int _76;
  int _77;
  long unsigned int _80;
  long unsigned int _81;
  long unsigned int _82;
  long unsigned int _92;
  long unsigned int _93;
  long unsigned int _97;
  long unsigned int _98;
  long unsigned int _102;
  long unsigned int _103;
  long unsigned int _112;

  __BB(2):
  state_87 = vstate_86(D);
  mt_88 = &state_87->mt;
  _1 = state_87->mti;
  if (_1 > 623)
    goto __BB3;
  else
    goto __BB19;

  __BB(3):
  kk_90 = 0;
  goto __BB8;

  __BB(4):
  _2 = (long unsigned int) kk_78;
  _3 = _2 * 8ul;
  _4 = mt_88 + _3;
  _5 = __MEM <long unsigned int> ((long unsigned int * const)_4);
  _6 = 2147483648ul;
  _7 = _5 & _6;
  _8 = (sizetype) kk_78;
  _9 = _8 + 1ul;
  _10 = _9 * 8ul;
  _11 = mt_88 + _10;
  _12 = __MEM <long unsigned int> ((long unsigned int * const)_11);
  _13 = 2147483647ul;
  _14 = _12 & _13;
  y_101 = _7 | _14;
  _15 = (sizetype) kk_78;
  _16 = _15 + 397ul;
  _17 = _16 * 8ul;
  _18 = mt_88 + _17;
  _19 = __MEM <long unsigned int> ((long unsigned int * const)_18);
  _20 = y_101 >> 1;
  _21 = _19 ^ _20;
  _22 = y_101 & 1ul;
  if (_22 != 0ul)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _103 = 2567483615ul;
  goto __BB7;

  __BB(6):
  _102 = 0ul;
  goto __BB7;

  __BB(7):
  _80 = __PHI (__BB5: _103, __BB6: _102);
  _23 = (long unsigned int) kk_78;
  _24 = _23 * 8ul;
  _25 = mt_88 + _24;
  _26 = _80 ^ _21;
  __MEM <long unsigned int> ((long unsigned int * const)_25) = _26;
  kk_105 = kk_78 + 1;
  goto __BB8;

  __BB(8,loop_header(1)):
  kk_78 = __PHI (__BB3: kk_90, __BB7: kk_105);
  if (kk_78 <= 226)
    goto __BB4;
  else
    goto __BB9;

  __BB(9):
  goto __BB14;

  __BB(10):
  _27 = (long unsigned int) kk_79;
  _28 = _27 * 8ul;
  _29 = mt_88 + _28;
  _30 = __MEM <long unsigned int> ((long unsigned int * const)_29);
  _31 = 2147483648ul;
  _32 = _30 & _31;
  _33 = (sizetype) kk_79;
  _34 = _33 + 1ul;
  _35 = _34 * 8ul;
  _36 = mt_88 + _35;
  _37 = __MEM <long unsigned int> ((long unsigned int * const)_36);
  _38 = 2147483647ul;
  _39 = _37 & _38;
  y_96 = _32 | _39;
  _40 = (long unsigned int) kk_79;
  _41 = _40 * 8ul;
  _42 = _41 + 18446744073709549800ul;
  _43 = mt_88 + _42;
  _44 = __MEM <long unsigned int> ((long unsigned int * const)_43);
  _45 = y_96 >> 1;
  _46 = _44 ^ _45;
  _47 = y_96 & 1ul;
  if (_47 != 0ul)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _98 = 2567483615ul;
  goto __BB13;

  __BB(12):
  _97 = 0ul;
  goto __BB13;

  __BB(13):
  _81 = __PHI (__BB11: _98, __BB12: _97);
  _48 = (long unsigned int) kk_79;
  _49 = _48 * 8ul;
  _50 = mt_88 + _49;
  _51 = _81 ^ _46;
  __MEM <long unsigned int> ((long unsigned int * const)_50) = _51;
  kk_100 = kk_79 + 1;
  goto __BB14;

  __BB(14,loop_header(2)):
  kk_79 = __PHI (__BB9: kk_78, __BB13: kk_100);
  if (kk_79 <= 622)
    goto __BB10;
  else
    goto __BB15;

  __BB(15):
  _52 = mt_88 + 4984ul;
  _53 = __MEM <long unsigned int> ((long unsigned int * const)_52);
  _54 = 2147483648ul;
  _55 = _53 & _54;
  _56 = __MEM <long unsigned int> (mt_88);
  _57 = 2147483647ul;
  _58 = _56 & _57;
  y_91 = _55 | _58;
  _59 = mt_88 + 3168ul;
  _60 = __MEM <long unsigned int> ((long unsigned int * const)_59);
  _61 = y_91 >> 1;
  _62 = _60 ^ _61;
  _63 = y_91 & 1ul;
  if (_63 != 0ul)
    goto __BB16;
  else
    goto __BB17;

  __BB(16):
  _93 = 2567483615ul;
  goto __BB18;

  __BB(17):
  _92 = 0ul;
  goto __BB18;

  __BB(18):
  _82 = __PHI (__BB16: _93, __BB17: _92);
  _64 = mt_88 + 4984ul;
  _65 = _82 ^ _62;
  __MEM <long unsigned int> ((long unsigned int * const)_64) = _65;
  state_87->mti = 0;
  goto __BB19;

  __BB(19):
  _66 = state_87->mti;
  _67 = (long unsigned int) _66;
  _68 = _67 * 8ul;
  _69 = mt_88 + _68;
  k_106 = __MEM <long unsigned int> ((long unsigned int * const)_69);
  _70 = k_106 >> 11;
  k_107 = k_106 ^ _70;
  _71 = k_107 << 7;
  _72 = _71 & 2636928640ul;
  k_108 = k_107 ^ _72;
  _73 = k_108 << 15;
  _74 = _73 & 4022730752ul;
  k_109 = k_108 ^ _74;
  _75 = k_109 >> 18;
  k_110 = k_109 ^ _75;
  _76 = state_87->mti;
  _77 = _76 + 1;
  state_87->mti = _77;
  _112 = k_110;
  goto __BB20;

  __BB(20):
L17:
  return _112;

}


double __GIMPLE (ssa)
mt_get_double (void * vstate)
{
  double D_3154;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = mt_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 4.294967296e+9;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
mt_set (void * vstate, long unsigned int s)
{
  int i;
  struct mt_state_t * state;
  long unsigned int _1;
  int _2;
  long unsigned int _3;
  int _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  long unsigned int _9;
  long unsigned int _10;
  long unsigned int _11;
  long unsigned int _12;

  __BB(2):
  state_17 = vstate_16(D);
  if (s_18(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_19 = 4357ul;
  goto __BB4;

  __BB(4):
  s_13 = __PHI (__BB2: s_18(D), __BB3: s_19);
  _1 = s_13 & 4294967295ul;
  state_17->mt[0] = _1;
  i_22 = 1;
  goto __BB6;

  __BB(5):
  _2 = i_14 + _Literal (int) -1;
  _3 = state_17->mt[_2];
  _4 = i_14 + _Literal (int) -1;
  _5 = state_17->mt[_4];
  _6 = _5 >> 30;
  _7 = _3 ^ _6;
  _8 = _7 * 1812433253ul;
  _9 = (long unsigned int) i_14;
  _10 = _8 + _9;
  state_17->mt[i_14] = _10;
  _11 = state_17->mt[i_14];
  _12 = _11 & 4294967295ul;
  state_17->mt[i_14] = _12;
  i_26 = i_14 + 1;
  goto __BB6;

  __BB(6,loop_header(1)):
  i_14 = __PHI (__BB4: i_22, __BB5: i_26);
  if (i_14 <= 623)
    goto __BB5;
  else
    goto __BB7;

  __BB(7):
  state_17->mti = i_14;
  return;

}


void __GIMPLE (ssa)
mt_1999_set (void * vstate, long unsigned int s)
{
  int i;
  struct mt_state_t * state;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  long unsigned int _9;

  __BB(2):
  state_15 = vstate_14(D);
  if (s_16(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_17 = 4357ul;
  goto __BB4;

  __BB(4):
  s_10 = __PHI (__BB2: s_16(D), __BB3: s_17);
  i_18 = 0;
  goto __BB6;

  __BB(5):
  _1 = s_11 & 4294901760ul;
  state_15->mt[i_12] = _1;
  _2 = s_11 * 69069ul;
  _3 = _2 + 1ul;
  s_22 = _3 & 4294967295ul;
  _4 = state_15->mt[i_12];
  _5 = s_22 >> 16;
  _6 = _5 & 65535ul;
  _7 = _4 | _6;
  state_15->mt[i_12] = _7;
  _8 = s_22 * 69069ul;
  _9 = _8 + 1ul;
  s_24 = _9 & 4294967295ul;
  i_25 = i_12 + 1;
  goto __BB6;

  __BB(6,loop_header(1)):
  s_11 = __PHI (__BB4: s_10, __BB5: s_24);
  i_12 = __PHI (__BB4: i_18, __BB5: i_25);
  if (i_12 <= 623)
    goto __BB5;
  else
    goto __BB7;

  __BB(7):
  state_15->mti = i_12;
  return;

}


void __GIMPLE (ssa)
mt_1998_set (void * vstate, long unsigned int s)
{
  int i;
  struct mt_state_t * state;
  long unsigned int _1;
  int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;

  __BB(2):
  state_10 = vstate_9(D);
  if (s_11(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_12 = 4357ul;
  goto __BB4;

  __BB(4):
  s_6 = __PHI (__BB2: s_11(D), __BB3: s_12);
  _1 = s_6 & 4294967295ul;
  state_10->mt[0] = _1;
  i_15 = 1;
  goto __BB6;

  __BB(5):
  _2 = i_7 + _Literal (int) -1;
  _3 = state_10->mt[_2];
  _4 = _3 * 69069ul;
  _5 = _4 & 4294967295ul;
  state_10->mt[i_7] = _5;
  i_18 = i_7 + 1;
  goto __BB6;

  __BB(6,loop_header(1)):
  i_7 = __PHI (__BB4: i_15, __BB5: i_18);
  if (i_7 <= 623)
    goto __BB5;
  else
    goto __BB7;

  __BB(7):
  state_10->mti = i_7;
  return;

}


