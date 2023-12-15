__attribute__((access ("^0[ ]^1[ ]^2[ ]", )))
int __GIMPLE (ssa)
gsl_poly_dd_init (double * dd, const double * xa, const double * ya, size_t size)
{
  size_t j;
  size_t i;
  int D_3164;
  double _1;
  long unsigned int _2;
  const double * _3;
  double _4;
  long unsigned int _5;
  sizetype _6;
  const double * _7;
  double _8;
  double _9;
  long unsigned int _10;
  const double * _11;
  double _12;
  long unsigned int _13;
  sizetype _14;
  const double * _15;
  double _16;
  double _17;
  long unsigned int _18;
  double * _19;
  double _20;
  long unsigned int _21;
  double * _22;
  double _23;
  long unsigned int _24;
  sizetype _25;
  double * _26;
  double _27;
  double _28;
  long unsigned int _29;
  const double * _30;
  double _31;
  long unsigned int _32;
  long unsigned int _33;
  const double * _34;
  double _35;
  double _36;
  long unsigned int _37;
  double * _38;
  double _39;
  int _53;

  __BB(2):
  _1 = __MEM <const double> (ya_47(D));
  __MEM <double> (dd_48(D)) = _1;
  j_51 = size_50(D) + 18446744073709551615ul;
  goto __BB4;

  __BB(3):
  _2 = j_41 * 8ul;
  _3 = ya_47(D) + _2;
  _4 = __MEM <const double> (_3);
  _5 = j_41 * 8ul;
  _6 = _5 + 18446744073709551608ul;
  _7 = ya_47(D) + _6;
  _8 = __MEM <const double> (_7);
  _9 = _4 - _8;
  _10 = j_41 * 8ul;
  _11 = xa_56(D) + _10;
  _12 = __MEM <const double> (_11);
  _13 = j_41 * 8ul;
  _14 = _13 + 18446744073709551608ul;
  _15 = xa_56(D) + _14;
  _16 = __MEM <const double> (_15);
  _17 = _12 - _16;
  _18 = j_41 * 8ul;
  _19 = dd_48(D) + _18;
  _20 = _9 / _17;
  __MEM <double> (_19) = _20;
  j_60 = j_41 + 18446744073709551615ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  j_41 = __PHI (__BB2: j_51, __BB3: j_60);
  if (j_41 != 0ul)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_52 = 2ul;
  goto __BB10;

  __BB(6):
  j_54 = size_50(D) + 18446744073709551615ul;
  goto __BB8;

  __BB(7):
  _21 = j_42 * 8ul;
  _22 = dd_48(D) + _21;
  _23 = __MEM <double> (_22);
  _24 = j_42 * 8ul;
  _25 = _24 + 18446744073709551608ul;
  _26 = dd_48(D) + _25;
  _27 = __MEM <double> (_26);
  _28 = _23 - _27;
  _29 = j_42 * 8ul;
  _30 = xa_56(D) + _29;
  _31 = __MEM <const double> (_30);
  _32 = j_42 - i_40;
  _33 = _32 * 8ul;
  _34 = xa_56(D) + _33;
  _35 = __MEM <const double> (_34);
  _36 = _31 - _35;
  _37 = j_42 * 8ul;
  _38 = dd_48(D) + _37;
  _39 = _28 / _36;
  __MEM <double> (_38) = _39;
  j_58 = j_42 + 18446744073709551615ul;
  goto __BB8;

  __BB(8,loop_header(3)):
  j_42 = __PHI (__BB6: j_54, __BB7: j_58);
  if (j_42 >= i_40)
    goto __BB7;
  else
    goto __BB9;

  __BB(9):
  i_55 = i_40 + 1ul;
  goto __BB10;

  __BB(10,loop_header(2)):
  i_40 = __PHI (__BB5: i_52, __BB9: i_55);
  if (i_40 < size_50(D))
    goto __BB6;
  else
    goto __BB11;

  __BB(11):
  _53 = 0;
  goto __BB12;

  __BB(12):
L9:
  return _53;

}


__attribute__((access ("^0[ ]^2[ ]^3[ ]^5[ ]", )))
int __GIMPLE (ssa)
gsl_poly_dd_taylor (double * c, double xp, const double * dd, const double * xa, size_t size, double * w)
{
  size_t j;
  size_t i;
  int D_3166;
  long unsigned int _1;
  double * _2;
  long unsigned int _3;
  double * _4;
  long unsigned int _5;
  sizetype _6;
  double * _7;
  double _8;
  sizetype _9;
  sizetype _10;
  double * _11;
  double _12;
  double _13;
  long unsigned int _14;
  long unsigned int _15;
  long unsigned int _16;
  const double * _17;
  double _18;
  double _19;
  long unsigned int _20;
  double * _21;
  double _22;
  long unsigned int _23;
  double * _24;
  double _25;
  sizetype _26;
  sizetype _27;
  double * _28;
  double _29;
  long unsigned int _30;
  long unsigned int _31;
  long unsigned int _32;
  const double * _33;
  double _34;
  double _35;
  double _36;
  long unsigned int _37;
  double * _38;
  double _39;
  long unsigned int _40;
  long unsigned int _41;
  long unsigned int _42;
  double * _43;
  double _44;
  long unsigned int _45;
  double * _46;
  double _47;
  long unsigned int _48;
  long unsigned int _49;
  sizetype _50;
  const double * _51;
  double _52;
  double _53;
  long unsigned int _54;
  long unsigned int _55;
  double * _56;
  double _57;
  long unsigned int _58;
  int _77;

  __BB(2):
  i_67 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = i_59 * 8ul;
  _2 = c_73(D) + _1;
  __MEM <double> (_2) = 0.0;
  _3 = i_59 * 8ul;
  _4 = w_70(D) + _3;
  __MEM <double> (_4) = 0.0;
  i_89 = i_59 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  i_59 = __PHI (__BB2: i_67, __BB3: i_89);
  if (i_59 < size_69(D))
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  _5 = size_69(D) * 8ul;
  _6 = _5 + 18446744073709551608ul;
  _7 = w_70(D) + _6;
  __MEM <double> (_7) = 1.0e+0;
  _8 = __MEM <const double> (dd_72(D));
  __MEM <double> (c_73(D)) = _8;
  i_75 = size_69(D) + 18446744073709551615ul;
  goto __BB12;

  __BB(6):
  _9 = i_76 + 1ul;
  _10 = _9 * 8ul;
  _11 = w_70(D) + _10;
  _12 = __MEM <double> (_11);
  _13 = -_12;
  _14 = size_69(D) - i_76;
  _15 = _14 + 18446744073709551614ul;
  _16 = _15 * 8ul;
  _17 = xa_78(D) + _16;
  _18 = __MEM <const double> (_17);
  _19 = _18 - xp_79(D);
  _20 = i_76 * 8ul;
  _21 = w_70(D) + _20;
  _22 = _13 * _19;
  __MEM <double> (_21) = _22;
  j_81 = i_76 + 1ul;
  goto __BB8;

  __BB(7):
  _23 = j_61 * 8ul;
  _24 = w_70(D) + _23;
  _25 = __MEM <double> (_24);
  _26 = j_61 + 1ul;
  _27 = _26 * 8ul;
  _28 = w_70(D) + _27;
  _29 = __MEM <double> (_28);
  _30 = size_69(D) - i_76;
  _31 = _30 + 18446744073709551614ul;
  _32 = _31 * 8ul;
  _33 = xa_78(D) + _32;
  _34 = __MEM <const double> (_33);
  _35 = _34 - xp_79(D);
  _36 = _29 * _35;
  _37 = j_61 * 8ul;
  _38 = w_70(D) + _37;
  _39 = _25 - _36;
  __MEM <double> (_38) = _39;
  j_86 = j_61 + 1ul;
  goto __BB8;

  __BB(8,loop_header(3)):
  j_61 = __PHI (__BB6: j_81, __BB7: j_86);
  _40 = size_69(D) + 18446744073709551615ul;
  if (j_61 < _40)
    goto __BB7;
  else
    goto __BB9;

  __BB(9):
  j_82 = i_76;
  goto __BB11;

  __BB(10):
  _41 = j_62 - i_76;
  _42 = _41 * 8ul;
  _43 = c_73(D) + _42;
  _44 = __MEM <double> (_43);
  _45 = j_62 * 8ul;
  _46 = w_70(D) + _45;
  _47 = __MEM <double> (_46);
  _48 = size_69(D) - i_76;
  _49 = _48 * 8ul;
  _50 = _49 + 18446744073709551608ul;
  _51 = dd_72(D) + _50;
  _52 = __MEM <const double> (_51);
  _53 = _47 * _52;
  _54 = j_62 - i_76;
  _55 = _54 * 8ul;
  _56 = c_73(D) + _55;
  _57 = _44 + _53;
  __MEM <double> (_56) = _57;
  j_84 = j_62 + 1ul;
  goto __BB11;

  __BB(11,loop_header(4)):
  j_62 = __PHI (__BB9: j_82, __BB10: j_84);
  if (j_62 < size_69(D))
    goto __BB10;
  else
    goto __BB12;

  __BB(12,loop_header(2)):
  i_60 = __PHI (__BB5: i_75, __BB11: i_76);
  _58 = i_60;
  i_76 = _58 + 18446744073709551615ul;
  if (_58 != 0ul)
    goto __BB6;
  else
    goto __BB13;

  __BB(13):
  _77 = 0;
  goto __BB14;

  __BB(14):
L12:
  return _77;

}


__attribute__((access ("^0[ ]^1[ ]^2[ ]^3[ ]^4[ ]", )))
int __GIMPLE (ssa)
gsl_poly_dd_hermite_init (double * dd, double * za, const double * xa, const double * ya, const double * dya, const size_t size)
{
  size_t j;
  size_t i;
  const size_t N;
  int D_3170;
  double _1;
  long unsigned int _2;
  const double * _3;
  long unsigned int _4;
  double * _5;
  double _6;
  long unsigned int _7;
  const double * _8;
  long unsigned int _9;
  sizetype _10;
  double * _11;
  double _12;
  long unsigned int _13;
  const double * _14;
  double _15;
  long unsigned int _16;
  sizetype _17;
  const double * _18;
  double _19;
  double _20;
  long unsigned int _21;
  const double * _22;
  double _23;
  long unsigned int _24;
  sizetype _25;
  const double * _26;
  double _27;
  double _28;
  long unsigned int _29;
  double * _30;
  double _31;
  long unsigned int _32;
  sizetype _33;
  const double * _34;
  long unsigned int _35;
  sizetype _36;
  double * _37;
  double _38;
  long unsigned int _39;
  sizetype _40;
  const double * _41;
  long unsigned int _42;
  sizetype _43;
  double * _44;
  double _45;
  long unsigned int _46;
  double * _47;
  double _48;
  long unsigned int _49;
  sizetype _50;
  double * _51;
  double _52;
  double _53;
  long unsigned int _54;
  double * _55;
  double _56;
  long unsigned int _57;
  long unsigned int _58;
  double * _59;
  double _60;
  double _61;
  long unsigned int _62;
  double * _63;
  double _64;
  int _82;

  __BB(2):
  N_73 = size_72(D) * 2ul;
  _1 = __MEM <const double> (ya_75(D));
  __MEM <double> (dd_76(D)) = _1;
  j_78 = 0ul;
  goto __BB6;

  __BB(3):
  _2 = j_66 * 8ul;
  _3 = xa_88(D) + _2;
  _4 = j_66 * 16ul;
  _5 = za_85(D) + _4;
  _6 = __MEM <const double> (_3);
  __MEM <double> (_5) = _6;
  _7 = j_66 * 8ul;
  _8 = xa_88(D) + _7;
  _9 = j_66 * 16ul;
  _10 = _9 + 8ul;
  _11 = za_85(D) + _10;
  _12 = __MEM <const double> (_8);
  __MEM <double> (_11) = _12;
  if (j_66 != 0ul)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  _13 = j_66 * 8ul;
  _14 = ya_75(D) + _13;
  _15 = __MEM <const double> (_14);
  _16 = j_66 * 8ul;
  _17 = _16 + 18446744073709551608ul;
  _18 = ya_75(D) + _17;
  _19 = __MEM <const double> (_18);
  _20 = _15 - _19;
  _21 = j_66 * 8ul;
  _22 = xa_88(D) + _21;
  _23 = __MEM <const double> (_22);
  _24 = j_66 * 8ul;
  _25 = _24 + 18446744073709551608ul;
  _26 = xa_88(D) + _25;
  _27 = __MEM <const double> (_26);
  _28 = _23 - _27;
  _29 = j_66 * 16ul;
  _30 = dd_76(D) + _29;
  _31 = _20 / _28;
  __MEM <double> (_30) = _31;
  _32 = j_66 * 8ul;
  _33 = _32 + 18446744073709551608ul;
  _34 = dya_79(D) + _33;
  _35 = j_66 * 16ul;
  _36 = _35 + 18446744073709551608ul;
  _37 = dd_76(D) + _36;
  _38 = __MEM <const double> (_34);
  __MEM <double> (_37) = _38;
  goto __BB5;

  __BB(5):
  j_93 = j_66 + 1ul;
  goto __BB6;

  __BB(6,loop_header(1)):
  j_66 = __PHI (__BB2: j_78, __BB5: j_93);
  if (j_66 < size_72(D))
    goto __BB3;
  else
    goto __BB7;

  __BB(7):
  _39 = size_72(D) * 8ul;
  _40 = _39 + 18446744073709551608ul;
  _41 = dya_79(D) + _40;
  _42 = N_73 * 8ul;
  _43 = _42 + 18446744073709551608ul;
  _44 = dd_76(D) + _43;
  _45 = __MEM <const double> (_41);
  __MEM <double> (_44) = _45;
  i_81 = 2ul;
  goto __BB12;

  __BB(8):
  j_83 = N_73 + 18446744073709551615ul;
  goto __BB10;

  __BB(9):
  _46 = j_67 * 8ul;
  _47 = dd_76(D) + _46;
  _48 = __MEM <double> (_47);
  _49 = j_67 * 8ul;
  _50 = _49 + 18446744073709551608ul;
  _51 = dd_76(D) + _50;
  _52 = __MEM <double> (_51);
  _53 = _48 - _52;
  _54 = j_67 * 8ul;
  _55 = za_85(D) + _54;
  _56 = __MEM <double> (_55);
  _57 = j_67 - i_65;
  _58 = _57 * 8ul;
  _59 = za_85(D) + _58;
  _60 = __MEM <double> (_59);
  _61 = _56 - _60;
  _62 = j_67 * 8ul;
  _63 = dd_76(D) + _62;
  _64 = _53 / _61;
  __MEM <double> (_63) = _64;
  j_87 = j_67 + 18446744073709551615ul;
  goto __BB10;

  __BB(10,loop_header(3)):
  j_67 = __PHI (__BB8: j_83, __BB9: j_87);
  if (j_67 >= i_65)
    goto __BB9;
  else
    goto __BB11;

  __BB(11):
  i_84 = i_65 + 1ul;
  goto __BB12;

  __BB(12,loop_header(2)):
  i_65 = __PHI (__BB7: i_81, __BB11: i_84);
  if (i_65 < N_73)
    goto __BB8;
  else
    goto __BB13;

  __BB(13):
  _82 = 0;
  goto __BB14;

  __BB(14):
L11:
  return _82;

}


