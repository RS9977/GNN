__attribute__((access ("^0[ ]^3[ ]", )))
int __GIMPLE (ssa)
gsl_poly_eval_derivs (const double * c, const size_t lenc, const double x, double * res, const size_t lenres)
{
  double f;
  size_t lmax;
  size_t l;
  size_t k;
  size_t nmax;
  size_t n;
  size_t i;
  int D_3133;
  size_t iftmp.0;
  long unsigned int _1;
  sizetype _2;
  const double * _3;
  long unsigned int _4;
  double * _5;
  double _6;
  long unsigned int _7;
  double * _8;
  long unsigned int _9;
  double _10;
  double _11;
  long unsigned int _12;
  sizetype _13;
  const double * _14;
  double _15;
  double _16;
  long unsigned int _17;
  double * _18;
  double _19;
  double _20;
  long unsigned int _21;
  sizetype _22;
  double * _23;
  double _24;
  long unsigned int _25;
  double * _26;
  double _27;
  long unsigned int _28;
  double _29;
  long unsigned int _30;
  double * _31;
  double _32;
  long unsigned int _33;
  double * _34;
  double _35;
  size_t _45;
  int _60;
  size_t _69;
  size_t _70;

  __BB(2):
  i_51 = 0ul;
  n_52 = 0ul;
  nmax_53 = 0ul;
  goto __BB7;

  __BB(3):
  if (n_40 < lenc_57(D))
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  _1 = lenc_57(D) * 8ul;
  _2 = _1 + 18446744073709551608ul;
  _3 = c_67(D) + _2;
  _4 = i_36 * 8ul;
  _5 = res_62(D) + _4;
  _6 = __MEM <const double> (_3);
  __MEM <double> (_5) = _6;
  nmax_78 = n_40;
  n_79 = n_40 + 1ul;
  goto __BB6;

  __BB(5):
  _7 = i_36 * 8ul;
  _8 = res_62(D) + _7;
  __MEM <double> (_8) = 0.0;
  goto __BB6;

  __BB(6):
  n_39 = __PHI (__BB4: n_79, __BB5: n_40);
  nmax_41 = __PHI (__BB4: nmax_78, __BB5: nmax_42);
  i_80 = i_36 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_36 = __PHI (__BB2: i_51, __BB6: i_80);
  n_40 = __PHI (__BB2: n_52, __BB6: n_39);
  nmax_42 = __PHI (__BB2: nmax_53, __BB6: nmax_41);
  if (i_36 < lenres_55(D))
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  i_56 = 0ul;
  goto __BB16;

  __BB(9):
  _9 = lenc_57(D) - i_37;
  k_65 = _9 + 18446744073709551615ul;
  _10 = __MEM <double> (res_62(D));
  _11 = x_66(D) * _10;
  _12 = k_65 * 8ul;
  _13 = _12 + 18446744073709551608ul;
  _14 = c_67(D) + _13;
  _15 = __MEM <const double> (_14);
  _16 = _11 + _15;
  __MEM <double> (res_62(D)) = _16;
  if (nmax_42 >= k_65)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  _70 = k_65 + 18446744073709551615ul;
  goto __BB12;

  __BB(11):
  _69 = nmax_42;
  goto __BB12;

  __BB(12):
  _45 = __PHI (__BB10: _70, __BB11: _69);
  lmax_71 = _45;
  l_72 = 1ul;
  goto __BB14;

  __BB(13):
  _17 = l_43 * 8ul;
  _18 = res_62(D) + _17;
  _19 = __MEM <double> (_18);
  _20 = x_66(D) * _19;
  _21 = l_43 * 8ul;
  _22 = _21 + 18446744073709551608ul;
  _23 = res_62(D) + _22;
  _24 = __MEM <double> (_23);
  _25 = l_43 * 8ul;
  _26 = res_62(D) + _25;
  _27 = _20 + _24;
  __MEM <double> (_26) = _27;
  l_75 = l_43 + 1ul;
  goto __BB14;

  __BB(14,loop_header(4)):
  l_43 = __PHI (__BB12: l_72, __BB13: l_75);
  if (l_43 <= lmax_71)
    goto __BB13;
  else
    goto __BB15;

  __BB(15):
  i_73 = i_37 + 1ul;
  goto __BB16;

  __BB(16,loop_header(2)):
  i_37 = __PHI (__BB8: i_56, __BB15: i_73);
  _28 = lenc_57(D) + 18446744073709551615ul;
  if (i_37 < _28)
    goto __BB9;
  else
    goto __BB17;

  __BB(17):
  f_58 = 1.0e+0;
  i_59 = 2ul;
  goto __BB19;

  __BB(18):
  _29 = (double) i_38;
  f_61 = f_44 * _29;
  _30 = i_38 * 8ul;
  _31 = res_62(D) + _30;
  _32 = __MEM <double> (_31);
  _33 = i_38 * 8ul;
  _34 = res_62(D) + _33;
  _35 = f_61 * _32;
  __MEM <double> (_34) = _35;
  i_64 = i_38 + 1ul;
  goto __BB19;

  __BB(19,loop_header(3)):
  i_38 = __PHI (__BB17: i_59, __BB18: i_64);
  f_44 = __PHI (__BB17: f_58, __BB18: f_61);
  if (i_38 <= nmax_42)
    goto __BB18;
  else
    goto __BB20;

  __BB(20):
  _60 = 0;
  goto __BB21;

  __BB(21):
L18:
  return _60;

}


