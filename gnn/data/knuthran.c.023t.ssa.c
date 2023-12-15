__attribute__((access ("^0[ ]^2[ ]", )))
void __GIMPLE (ssa)
ran_array (long unsigned int * aa, unsigned int n, long unsigned int * ran_x)
{
  unsigned int j;
  unsigned int i;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int * _3;
  long unsigned int _4;
  long unsigned int _5;
  long unsigned int * _6;
  long unsigned int _7;
  unsigned int _8;
  long unsigned int _9;
  long unsigned int _10;
  long unsigned int * _11;
  long unsigned int _12;
  unsigned int _13;
  long unsigned int _14;
  long unsigned int _15;
  long unsigned int * _16;
  long unsigned int _17;
  long unsigned int _18;
  long unsigned int _19;
  long unsigned int _20;
  long unsigned int * _21;
  long unsigned int _22;
  unsigned int _23;
  long unsigned int _24;
  long unsigned int _25;
  long unsigned int * _26;
  long unsigned int _27;
  unsigned int _28;
  long unsigned int _29;
  long unsigned int _30;
  long unsigned int * _31;
  long unsigned int _32;
  long unsigned int _33;
  long unsigned int _34;
  long unsigned int _35;
  long unsigned int * _36;
  long unsigned int _37;
  unsigned int _38;
  long unsigned int _39;
  long unsigned int _40;
  long unsigned int * _41;
  long unsigned int _42;
  unsigned int _43;
  long unsigned int _44;
  long unsigned int _45;
  long unsigned int * _46;
  long unsigned int _47;
  long unsigned int _48;
  long unsigned int _49;
  long unsigned int _50;
  long unsigned int * _51;
  long unsigned int _52;

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
  _7 = __MEM <long unsigned int> (_3);
  __MEM <long unsigned int> (_6) = _7;
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
  _12 = __MEM <long unsigned int> (_11);
  _13 = j_56 + 4294967259u;
  _14 = (long unsigned int) _13;
  _15 = _14 * 8ul;
  _16 = aa_67(D) + _15;
  _17 = __MEM <long unsigned int> (_16);
  _18 = _12 - _17;
  _19 = (long unsigned int) j_56;
  _20 = _19 * 8ul;
  _21 = aa_67(D) + _20;
  _22 = _18 & 1073741823ul;
  __MEM <long unsigned int> (_21) = _22;
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
  _27 = __MEM <long unsigned int> (_26);
  _28 = j_57 + 4294967259u;
  _29 = (long unsigned int) _28;
  _30 = _29 * 8ul;
  _31 = aa_67(D) + _30;
  _32 = __MEM <long unsigned int> (_31);
  _33 = _27 - _32;
  _34 = (long unsigned int) i_53;
  _35 = _34 * 8ul;
  _36 = ran_x_68(D) + _35;
  _37 = _33 & 1073741823ul;
  __MEM <long unsigned int> (_36) = _37;
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
  _42 = __MEM <long unsigned int> (_41);
  _43 = i_54 + 4294967259u;
  _44 = (long unsigned int) _43;
  _45 = _44 * 8ul;
  _46 = ran_x_68(D) + _45;
  _47 = __MEM <long unsigned int> (_46);
  _48 = _42 - _47;
  _49 = (long unsigned int) i_54;
  _50 = _49 * 8ul;
  _51 = ran_x_68(D) + _50;
  _52 = _48 & 1073741823ul;
  __MEM <long unsigned int> (_51) = _52;
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
  unsigned int i;
  struct ran_state_t * state;
  long unsigned int D_3180;
  long unsigned int[100] * _1;
  long unsigned int[2009] * _2;
  unsigned int _3;
  unsigned int _4;
  long unsigned int _12;

  __BB(2):
  state_7 = vstate_6(D);
  i_9 = state_7->i;
  if (i_9 == 0u)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _1 = &state_7->ran_x;
  _2 = &state_7->aa;
  ran_array (_2, 2009u, _1);
  goto __BB4;

  __BB(4):
  _3 = i_9 + 1u;
  _4 = _3 % 2009u;
  state_7->i = _4;
  _12 = state_7->aa[i_9];
  goto __BB5;

  __BB(5):
L2:
  return _12;

}


double __GIMPLE (ssa)
ran_get_double (void * vstate)
{
  struct ran_state_t * state;
  double D_3176;
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
  long int _8;
  int _9;
  long int _10;
  long int _11;
  long int _12;
  int _13;
  long int _14;
  long int _15;
  long int _16;
  int _17;
  long int _18;
  int _19;
  long int _20;
  long int _21;
  long int _22;
  int _23;
  long int _24;
  long int _25;
  int _26;
  long int _27;
  long int _28;
  long int _29;
  long int _30;
  long int _31;
  long int _32;
  long int _33;
  long int _34;
  long int _35;
  int _36;
  long unsigned int _37;
  long int _38;
  int _39;
  long unsigned int _40;

  __BB(2):
  state_65 = vstate_64(D);
  _1 = s_66(D) + 2ul;
  _2 = (long int) _1;
  ss_67 = _2 & 1073741822l;
  j_68 = 0;
  goto __BB6;

  __BB(3):
  x[j_41] = ss_51;
  ss_100 = ss_51 << 1;
  if (ss_100 > 1073741823l)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  ss_101 = ss_100 + _Literal (long int) -1073741822l;
  goto __BB5;

  __BB(5):
  ss_50 = __PHI (__BB3: ss_100, __BB4: ss_101);
  j_102 = j_41 + 1;
  goto __BB6;

  __BB(6,loop_header(1)):
  j_41 = __PHI (__BB2: j_68, __BB5: j_102);
  ss_51 = __PHI (__BB2: ss_67, __BB5: ss_50);
  if (j_41 <= 99)
    goto __BB3;
  else
    goto __BB7;

  __BB(7):
  goto __BB9;

  __BB(8):
  x[j_42] = 0l;
  j_98 = j_42 + 1;
  goto __BB9;

  __BB(9,loop_header(2)):
  j_42 = __PHI (__BB7: j_41, __BB8: j_98);
  if (j_42 <= 198)
    goto __BB8;
  else
    goto __BB10;

  __BB(10):
  _3 = x[1];
  _4 = _3 + 1l;
  x[1] = _4;
  _5 = (long int) s_66(D);
  ss_71 = _5 & 1073741823l;
  t_72 = 69;
  goto __BB31;

  __BB(11):
  j_80 = 99;
  goto __BB13;

  __BB(12):
  _6 = j_43 * 2;
  _7 = x[j_43];
  x[_6] = _7;
  j_96 = j_43 + _Literal (int) -1;
  goto __BB13;

  __BB(13,loop_header(6)):
  j_43 = __PHI (__BB11: j_80, __BB12: j_96);
  if (j_43 > 0)
    goto __BB12;
  else
    goto __BB14;

  __BB(14):
  j_81 = 198;
  goto __BB16;

  __BB(15):
  _8 = x[j_44];
  _9 = 199 - j_44;
  _10 = _8 & 1073741822l;
  x[_9] = _10;
  j_94 = j_44 + _Literal (int) -2;
  goto __BB16;

  __BB(16,loop_header(7)):
  j_44 = __PHI (__BB14: j_81, __BB15: j_94);
  if (j_44 > 63)
    goto __BB15;
  else
    goto __BB17;

  __BB(17):
  j_82 = 198;
  goto __BB21;

  __BB(18):
  _11 = x[j_45];
  _12 = _11 & 1l;
  if (_12 != 0l)
    goto __BB19;
  else
    goto __BB20;

  __BB(19):
  _13 = j_45 + _Literal (int) -63;
  _14 = x[_13];
  _15 = x[j_45];
  _16 = _14 - _15;
  _17 = j_45 + _Literal (int) -63;
  _18 = _16 & 1073741823l;
  x[_17] = _18;
  _19 = j_45 + _Literal (int) -100;
  _20 = x[_19];
  _21 = x[j_45];
  _22 = _20 - _21;
  _23 = j_45 + _Literal (int) -100;
  _24 = _22 & 1073741823l;
  x[_23] = _24;
  goto __BB20;

  __BB(20):
  j_92 = j_45 + _Literal (int) -1;
  goto __BB21;

  __BB(21,loop_header(8)):
  j_45 = __PHI (__BB17: j_82, __BB20: j_92);
  if (j_45 > 99)
    goto __BB18;
  else
    goto __BB22;

  __BB(22):
  _25 = ss_52 & 1l;
  if (_25 != 0l)
    goto __BB23;
  else
    goto __BB28;

  __BB(23):
  j_83 = 100;
  goto __BB25;

  __BB(24):
  _26 = j_46 + _Literal (int) -1;
  _27 = x[_26];
  x[j_46] = _27;
  j_87 = j_46 + _Literal (int) -1;
  goto __BB25;

  __BB(25,loop_header(9)):
  j_46 = __PHI (__BB23: j_83, __BB24: j_87);
  if (j_46 > 0)
    goto __BB24;
  else
    goto __BB26;

  __BB(26):
  _28 = x[100];
  x[0] = _28;
  _29 = x[100];
  _30 = _29 & 1l;
  if (_30 != 0l)
    goto __BB27;
  else
    goto __BB28;

  __BB(27):
  _31 = x[37];
  _32 = x[100];
  _33 = _31 - _32;
  _34 = _33 & 1073741823l;
  x[37] = _34;
  goto __BB28;

  __BB(28):
  if (ss_52 != 0l)
    goto __BB29;
  else
    goto __BB30;

  __BB(29):
  ss_89 = ss_52 >> 1;
  goto __BB31;

  __BB(30):
  t_88 = t_49 + _Literal (int) -1;
  goto __BB31;

  __BB(31,loop_header(3)):
  t_49 = __PHI (__BB10: t_72, __BB29: t_49, __BB30: t_88);
  ss_52 = __PHI (__BB10: ss_71, __BB29: ss_89, __BB30: ss_52);
  if (t_49 != 0)
    goto __BB11;
  else
    goto __BB32;

  __BB(32):
  state_65->i = 0u;
  j_74 = 0;
  goto __BB34;

  __BB(33):
  _35 = x[j_47];
  _36 = j_47 + 63;
  _37 = (long unsigned int) _35;
  state_65->ran_x[_36] = _37;
  j_79 = j_47 + 1;
  goto __BB34;

  __BB(34,loop_header(4)):
  j_47 = __PHI (__BB32: j_74, __BB33: j_79);
  if (j_47 <= 36)
    goto __BB33;
  else
    goto __BB35;

  __BB(35):
  goto __BB37;

  __BB(36):
  _38 = x[j_48];
  _39 = j_48 + _Literal (int) -37;
  _40 = (long unsigned int) _38;
  state_65->ran_x[_39] = _40;
  j_77 = j_48 + 1;
  goto __BB37;

  __BB(37,loop_header(5)):
  j_48 = __PHI (__BB35: j_47, __BB36: j_77);
  if (j_48 <= 99)
    goto __BB36;
  else
    goto __BB38;

  __BB(38):
  goto __BB39;

  __BB(39):
  x ={v} _Literal (long int[199]) {CLOBBER};
  goto __BB40;

  __BB(40):
L39:
  return;

}


