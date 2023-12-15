__attribute__((access ("^0[ ]^2[ ]", )))
void __GIMPLE (ssa)
ran_array (long int * aa, unsigned int n, long int * ran_x)
{
  unsigned int j;
  unsigned int i;
  long unsigned int _1;
  long unsigned int _2;
  long int * _3;
  long unsigned int _4;
  long unsigned int _5;
  long int * _6;
  long int _7;
  unsigned int _8;
  long unsigned int _9;
  long unsigned int _10;
  long int * _11;
  long int _12;
  unsigned int _13;
  long unsigned int _14;
  long unsigned int _15;
  long int * _16;
  long int _17;
  long int _18;
  long unsigned int _19;
  long unsigned int _20;
  long int * _21;
  long int _22;
  unsigned int _23;
  long unsigned int _24;
  long unsigned int _25;
  long int * _26;
  long int _27;
  unsigned int _28;
  long unsigned int _29;
  long unsigned int _30;
  long int * _31;
  long int _32;
  long int _33;
  long unsigned int _34;
  long unsigned int _35;
  long int * _36;
  long int _37;
  unsigned int _38;
  long unsigned int _39;
  long unsigned int _40;
  long int * _41;
  long int _42;
  unsigned int _43;
  long unsigned int _44;
  long unsigned int _45;
  long int * _46;
  long int _47;
  long int _48;
  long unsigned int _49;
  long unsigned int _50;
  long int * _51;
  long int _52;

  __BB(2):
  j_63 = 0u;
  goto __BB4;

  __BB(3):
  _1 = (long unsigned int) j_55;
  _2 = _1 * 8ul;
  _3 = ran_x_68(D) + _2;
  _4 = (long unsigned int) j_55;
  _5 = _4 * 8ul;
  _6 = aa_67(D) + _5;
  _7 = __MEM <long int> (_3);
  __MEM <long int> (_6) = _7;
  j_78 = j_55 + 1u;
  goto __BB4;

  __BB(4,loop_header(1)):
  j_55 = __PHI (__BB2: j_63, __BB3: j_78);
  if (j_55 <= 99u)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  goto __BB7;

  __BB(6):
  _8 = j_56 + 4294967196u;
  _9 = (long unsigned int) _8;
  _10 = _9 * 8ul;
  _11 = aa_67(D) + _10;
  _12 = __MEM <long int> (_11);
  _13 = j_56 + 4294967259u;
  _14 = (long unsigned int) _13;
  _15 = _14 * 8ul;
  _16 = aa_67(D) + _15;
  _17 = __MEM <long int> (_16);
  _18 = _12 - _17;
  _19 = (long unsigned int) j_56;
  _20 = _19 * 8ul;
  _21 = aa_67(D) + _20;
  _22 = _18 & 1073741823l;
  __MEM <long int> (_21) = _22;
  j_76 = j_56 + 1u;
  goto __BB7;

  __BB(7,loop_header(2)):
  j_56 = __PHI (__BB5: j_55, __BB6: j_76);
  if (j_56 < n_65(D))
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  i_66 = 0u;
  goto __BB10;

  __BB(9):
  _23 = j_57 + 4294967196u;
  _24 = (long unsigned int) _23;
  _25 = _24 * 8ul;
  _26 = aa_67(D) + _25;
  _27 = __MEM <long int> (_26);
  _28 = j_57 + 4294967259u;
  _29 = (long unsigned int) _28;
  _30 = _29 * 8ul;
  _31 = aa_67(D) + _30;
  _32 = __MEM <long int> (_31);
  _33 = _27 - _32;
  _34 = (long unsigned int) i_53;
  _35 = _34 * 8ul;
  _36 = ran_x_68(D) + _35;
  _37 = _33 & 1073741823l;
  __MEM <long int> (_36) = _37;
  i_73 = i_53 + 1u;
  j_74 = j_57 + 1u;
  goto __BB10;

  __BB(10,loop_header(3)):
  i_53 = __PHI (__BB8: i_66, __BB9: i_73);
  j_57 = __PHI (__BB8: j_56, __BB9: j_74);
  if (i_53 <= 36u)
    goto __BB9;
  else
    goto __BB11;

  __BB(11):
  goto __BB13;

  __BB(12):
  _38 = j_58 + 4294967196u;
  _39 = (long unsigned int) _38;
  _40 = _39 * 8ul;
  _41 = aa_67(D) + _40;
  _42 = __MEM <long int> (_41);
  _43 = i_54 + 4294967259u;
  _44 = (long unsigned int) _43;
  _45 = _44 * 8ul;
  _46 = ran_x_68(D) + _45;
  _47 = __MEM <long int> (_46);
  _48 = _42 - _47;
  _49 = (long unsigned int) i_54;
  _50 = _49 * 8ul;
  _51 = ran_x_68(D) + _50;
  _52 = _48 & 1073741823l;
  __MEM <long int> (_51) = _52;
  i_70 = i_54 + 1u;
  j_71 = j_58 + 1u;
  goto __BB13;

  __BB(13,loop_header(4)):
  i_54 = __PHI (__BB11: i_53, __BB12: i_70);
  j_58 = __PHI (__BB11: j_57, __BB12: j_71);
  if (i_54 <= 99u)
    goto __BB12;
  else
    goto __BB14;

  __BB(14):
  return;

}


long unsigned int __GIMPLE (ssa)
ran_get (void * vstate)
{
  long unsigned int v;
  unsigned int i;
  struct ran_state_t * state;
  long unsigned int D_3177;
  long int[100] * _1;
  long int[1009] * _2;
  long int _3;
  unsigned int _4;
  unsigned int _5;
  long unsigned int _14;

  __BB(2):
  state_8 = vstate_7(D);
  i_10 = state_8->i;
  if (i_10 == 0u)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _1 = &state_8->ran_x;
  _2 = &state_8->aa;
  ran_array (_2, 1009u, _1);
  goto __BB4;

  __BB(4):
  _3 = state_8->aa[i_10];
  v_12 = (long unsigned int) _3;
  _4 = i_10 + 1u;
  _5 = _4 % 100u;
  state_8->i = _5;
  _14 = v_12;
  goto __BB5;

  __BB(5):
L2:
  return _14;

}


double __GIMPLE (ssa)
ran_get_double (void * vstate)
{
  struct ran_state_t * state;
  double D_3173;
  long unsigned int _1;
  double _2;
  double _7;

  __BB(2):
  state_4 = vstate_3(D);
  _1 = ran_get (state_4);
  _2 = (double) _1;
  _7 = _2 / 1.073741824e+9;
  goto __BB3;

  __BB(3):
L0:
  return _7;

}


void __GIMPLE (ssa)
ran_set (void * vstate, long unsigned int s)
{
  register long int ss;
  register int t;
  register int j;
  long int x[199];
  struct ran_state_t * state;
  long unsigned int _1;
  long int _2;
  long int _3;
  long int _4;
  long int _5;
  int _6;
  long int _7;
  int _8;
  int _9;
  int _10;
  long int _11;
  long int _12;
  long int _13;
  int _14;
  long int _15;
  int _16;
  long int _17;
  long int _18;
  long int _19;
  int _20;
  long int _21;
  long int _22;
  int _23;
  long int _24;
  long int _25;
  long int _26;
  long int _27;
  long int _28;
  long int _29;
  int _30;
  long int _31;
  int _32;
  long int _33;
  long int[100] * _34;

  __BB(2):
  state_57 = vstate_56(D);
  if (s_58(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_59 = 314159ul;
  goto __BB4;

  __BB(4):
  s_35 = __PHI (__BB2: s_58(D), __BB3: s_59);
  _1 = s_35 + 2ul;
  _2 = (long int) _1;
  ss_60 = _2 & 1073741822l;
  j_61 = 0;
  goto __BB8;

  __BB(5):
  x[j_36] = ss_45;
  ss_92 = ss_45 << 1;
  if (ss_92 > 1073741823l)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  ss_93 = ss_92 + _Literal (long int) -1073741822l;
  goto __BB7;

  __BB(7):
  ss_44 = __PHI (__BB5: ss_92, __BB6: ss_93);
  j_94 = j_36 + 1;
  goto __BB8;

  __BB(8,loop_header(1)):
  j_36 = __PHI (__BB4: j_61, __BB7: j_94);
  ss_45 = __PHI (__BB4: ss_60, __BB7: ss_44);
  if (j_36 <= 99)
    goto __BB5;
  else
    goto __BB9;

  __BB(9):
  _3 = x[1];
  _4 = _3 + 1l;
  x[1] = _4;
  _5 = (long int) s_35;
  ss_64 = _5 & 1073741823l;
  t_65 = 69;
  goto __BB24;

  __BB(10):
  j_76 = 99;
  goto __BB12;

  __BB(11):
  _6 = j_37 * 2;
  _7 = x[j_37];
  x[_6] = _7;
  _8 = j_37 * 2;
  _9 = _8 + _Literal (int) -1;
  x[_9] = 0l;
  j_90 = j_37 + _Literal (int) -1;
  goto __BB12;

  __BB(12,loop_header(6)):
  j_37 = __PHI (__BB10: j_76, __BB11: j_90);
  if (j_37 > 0)
    goto __BB11;
  else
    goto __BB13;

  __BB(13):
  j_77 = 198;
  goto __BB15;

  __BB(14):
  _10 = j_38 + _Literal (int) -63;
  _11 = x[_10];
  _12 = x[j_38];
  _13 = _11 - _12;
  _14 = j_38 + _Literal (int) -63;
  _15 = _13 & 1073741823l;
  x[_14] = _15;
  _16 = j_38 + _Literal (int) -100;
  _17 = x[_16];
  _18 = x[j_38];
  _19 = _17 - _18;
  _20 = j_38 + _Literal (int) -100;
  _21 = _19 & 1073741823l;
  x[_20] = _21;
  j_87 = j_38 + _Literal (int) -1;
  goto __BB15;

  __BB(15,loop_header(7)):
  j_38 = __PHI (__BB13: j_77, __BB14: j_87);
  if (j_38 > 99)
    goto __BB14;
  else
    goto __BB16;

  __BB(16):
  _22 = ss_46 & 1l;
  if (_22 != 0l)
    goto __BB17;
  else
    goto __BB21;

  __BB(17):
  j_78 = 100;
  goto __BB19;

  __BB(18):
  _23 = j_39 + _Literal (int) -1;
  _24 = x[_23];
  x[j_39] = _24;
  j_82 = j_39 + _Literal (int) -1;
  goto __BB19;

  __BB(19,loop_header(8)):
  j_39 = __PHI (__BB17: j_78, __BB18: j_82);
  if (j_39 > 0)
    goto __BB18;
  else
    goto __BB20;

  __BB(20):
  _25 = x[100];
  x[0] = _25;
  _26 = x[37];
  _27 = x[100];
  _28 = _26 - _27;
  _29 = _28 & 1073741823l;
  x[37] = _29;
  goto __BB21;

  __BB(21):
  if (ss_46 != 0l)
    goto __BB22;
  else
    goto __BB23;

  __BB(22):
  ss_84 = ss_46 >> 1;
  goto __BB24;

  __BB(23):
  t_83 = t_43 + _Literal (int) -1;
  goto __BB24;

  __BB(24,loop_header(2)):
  t_43 = __PHI (__BB9: t_65, __BB22: t_43, __BB23: t_83);
  ss_46 = __PHI (__BB9: ss_64, __BB22: ss_84, __BB23: ss_46);
  if (t_43 != 0)
    goto __BB10;
  else
    goto __BB25;

  __BB(25):
  j_66 = 0;
  goto __BB27;

  __BB(26):
  _30 = j_40 + 63;
  _31 = x[j_40];
  state_57->ran_x[_30] = _31;
  j_75 = j_40 + 1;
  goto __BB27;

  __BB(27,loop_header(3)):
  j_40 = __PHI (__BB25: j_66, __BB26: j_75);
  if (j_40 <= 36)
    goto __BB26;
  else
    goto __BB28;

  __BB(28):
  goto __BB30;

  __BB(29):
  _32 = j_41 + _Literal (int) -37;
  _33 = x[j_41];
  state_57->ran_x[_32] = _33;
  j_73 = j_41 + 1;
  goto __BB30;

  __BB(30,loop_header(4)):
  j_41 = __PHI (__BB28: j_40, __BB29: j_73);
  if (j_41 <= 99)
    goto __BB29;
  else
    goto __BB31;

  __BB(31):
  j_67 = 0;
  goto __BB33;

  __BB(32):
  _34 = &state_57->ran_x;
  ran_array (&x, 199u, _34);
  j_71 = j_42 + 1;
  goto __BB33;

  __BB(33,loop_header(5)):
  j_42 = __PHI (__BB31: j_67, __BB32: j_71);
  if (j_42 <= 9)
    goto __BB32;
  else
    goto __BB34;

  __BB(34):
  state_57->i = 0u;
  goto __BB35;

  __BB(35):
  x ={v} _Literal (long int[199]) {CLOBBER};
  goto __BB36;

  __BB(36):
L34:
  return;

}


