int __GIMPLE (ssa)
gsl_fit_linear (const double * x, const size_t xstride, const double * y, const size_t ystride, const size_t n, double * c0, double * c1, double * cov_00, double * cov_01, double * cov_11, double * sumsq)
{
  const double d;
  const double dy;
  const double dx;
  double a;
  double b;
  double d2;
  double s2;
  const double dy;
  const double dx;
  size_t i;
  double m_dxdy;
  double m_dx2;
  double m_y;
  double m_x;
  int D_4404;
  long unsigned int _1;
  long unsigned int _2;
  const double * _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  long unsigned int _9;
  long unsigned int _10;
  const double * _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  long unsigned int _17;
  long unsigned int _18;
  const double * _19;
  double _20;
  long unsigned int _21;
  long unsigned int _22;
  const double * _23;
  double _24;
  double _25;
  double _26;
  double _27;
  double _28;
  double _29;
  double _30;
  double _31;
  double _32;
  double _33;
  double _34;
  double _35;
  long unsigned int _36;
  long unsigned int _37;
  const double * _38;
  double _39;
  long unsigned int _40;
  long unsigned int _41;
  const double * _42;
  double _43;
  double _44;
  double _45;
  double _46;
  double _47;
  double _48;
  double _49;
  double _50;
  double _51;
  double _52;
  double _53;
  double _54;
  double _55;
  double _56;
  double _57;
  double _58;
  double _59;
  double _60;
  double _61;
  double _62;
  int _97;

  __BB(2):
  m_x_71 = 0.0;
  m_y_72 = 0.0;
  m_dx2_73 = 0.0;
  m_dxdy_74 = 0.0;
  i_75 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = i_67 * xstride_98(D);
  _2 = _1 * 8ul;
  _3 = x_99(D) + _2;
  _4 = __MEM <const double> (_3);
  _5 = _4 - m_x_63;
  _6 = (double) i_67;
  _7 = _6 + 1.0e+0;
  _8 = _5 / _7;
  m_x_112 = m_x_63 + _8;
  _9 = i_67 * ystride_101(D);
  _10 = _9 * 8ul;
  _11 = y_102(D) + _10;
  _12 = __MEM <const double> (_11);
  _13 = _12 - m_y_64;
  _14 = (double) i_67;
  _15 = _14 + 1.0e+0;
  _16 = _13 / _15;
  m_y_113 = m_y_64 + _16;
  i_114 = i_67 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  m_x_63 = __PHI (__BB2: m_x_71, __BB3: m_x_112);
  m_y_64 = __PHI (__BB2: m_y_72, __BB3: m_y_113);
  i_67 = __PHI (__BB2: i_75, __BB3: i_114);
  if (i_67 < n_76(D))
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_77 = 0ul;
  goto __BB7;

  __BB(6):
  _17 = i_68 * xstride_98(D);
  _18 = _17 * 8ul;
  _19 = x_99(D) + _18;
  _20 = __MEM <const double> (_19);
  dx_107 = _20 - m_x_63;
  _21 = i_68 * ystride_101(D);
  _22 = _21 * 8ul;
  _23 = y_102(D) + _22;
  _24 = __MEM <const double> (_23);
  dy_108 = _24 - m_y_64;
  _25 = dx_107 * dx_107;
  _26 = _25 - m_dx2_65;
  _27 = (double) i_68;
  _28 = _27 + 1.0e+0;
  _29 = _26 / _28;
  m_dx2_109 = m_dx2_65 + _29;
  _30 = dx_107 * dy_108;
  _31 = _30 - m_dxdy_66;
  _32 = (double) i_68;
  _33 = _32 + 1.0e+0;
  _34 = _31 / _33;
  m_dxdy_110 = m_dxdy_66 + _34;
  i_111 = i_68 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  m_dx2_65 = __PHI (__BB5: m_dx2_73, __BB6: m_dx2_109);
  m_dxdy_66 = __PHI (__BB5: m_dxdy_74, __BB6: m_dxdy_110);
  i_68 = __PHI (__BB5: i_77, __BB6: i_111);
  if (i_68 < n_76(D))
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  s2_78 = 0.0;
  d2_79 = 0.0;
  b_80 = m_dxdy_66 / m_dx2_65;
  _35 = m_x_63 * b_80;
  a_81 = m_y_64 - _35;
  __MEM <double> (c0_83(D)) = a_81;
  __MEM <double> (c1_85(D)) = b_80;
  i_87 = 0ul;
  goto __BB10;

  __BB(9):
  _36 = i_69 * xstride_98(D);
  _37 = _36 * 8ul;
  _38 = x_99(D) + _37;
  _39 = __MEM <const double> (_38);
  dx_100 = _39 - m_x_63;
  _40 = i_69 * ystride_101(D);
  _41 = _40 * 8ul;
  _42 = y_102(D) + _41;
  _43 = __MEM <const double> (_42);
  dy_103 = _43 - m_y_64;
  _44 = b_80 * dx_100;
  d_104 = dy_103 - _44;
  _45 = d_104 * d_104;
  d2_105 = d2_70 + _45;
  i_106 = i_69 + 1ul;
  goto __BB10;

  __BB(10,loop_header(3)):
  i_69 = __PHI (__BB8: i_87, __BB9: i_106);
  d2_70 = __PHI (__BB8: d2_79, __BB9: d2_105);
  if (i_69 < n_76(D))
    goto __BB9;
  else
    goto __BB11;

  __BB(11):
  _46 = (double) n_76(D);
  _47 = _46 - 2.0e+0;
  s2_88 = d2_70 / _47;
  _48 = (double) n_76(D);
  _49 = 1.0e+0 / _48;
  _50 = s2_88 * _49;
  _51 = m_x_63 * m_x_63;
  _52 = _51 / m_dx2_65;
  _53 = _52 + 1.0e+0;
  _54 = _50 * _53;
  __MEM <double> (cov_00_89(D)) = _54;
  _55 = (double) n_76(D);
  _56 = m_dx2_65 * _55;
  _57 = s2_88 / _56;
  __MEM <double> (cov_11_91(D)) = _57;
  _58 = -m_x_63;
  _59 = s2_88 * _58;
  _60 = (double) n_76(D);
  _61 = m_dx2_65 * _60;
  _62 = _59 / _61;
  __MEM <double> (cov_01_93(D)) = _62;
  __MEM <double> (sumsq_95(D)) = d2_70;
  _97 = 0;
  goto __BB12;

  __BB(12):
L9:
  return _97;

}


int __GIMPLE (ssa)
gsl_fit_wlinear (const double * x, const size_t xstride, const double * w, const size_t wstride, const double * y, const size_t ystride, const size_t n, double * c0, double * c1, double * cov_00, double * cov_01, double * cov_11, double * chisq)
{
  const double d;
  const double dy;
  const double dx;
  const double wi;
  double a;
  double b;
  double d2;
  const double dy;
  const double dx;
  const double wi;
  const double wi;
  size_t i;
  double wm_dxdy;
  double wm_dx2;
  double wm_y;
  double wm_x;
  double W;
  int D_4412;
  long unsigned int _1;
  long unsigned int _2;
  const double * _3;
  long unsigned int _4;
  long unsigned int _5;
  const double * _6;
  double _7;
  double _8;
  double _9;
  double _10;
  long unsigned int _11;
  long unsigned int _12;
  const double * _13;
  double _14;
  double _15;
  double _16;
  double _17;
  long unsigned int _18;
  long unsigned int _19;
  const double * _20;
  long unsigned int _21;
  long unsigned int _22;
  const double * _23;
  double _24;
  long unsigned int _25;
  long unsigned int _26;
  const double * _27;
  double _28;
  double _29;
  double _30;
  double _31;
  double _32;
  double _33;
  double _34;
  double _35;
  double _36;
  double _37;
  double _38;
  double _39;
  double _40;
  double _41;
  double _42;
  double _43;
  double _44;
  double _45;
  double _46;
  double _47;
  long unsigned int _48;
  long unsigned int _49;
  const double * _50;
  long unsigned int _51;
  long unsigned int _52;
  const double * _53;
  double _54;
  long unsigned int _55;
  long unsigned int _56;
  const double * _57;
  double _58;
  double _59;
  double _60;
  double _61;
  int _105;

  __BB(2):
  W_79 = 0.0;
  wm_x_80 = 0.0;
  wm_y_81 = 0.0;
  wm_dx2_82 = 0.0;
  wm_dxdy_83 = 0.0;
  i_84 = 0ul;
  goto __BB6;

  __BB(3):
  _1 = i_74 * wstride_106(D);
  _2 = _1 * 8ul;
  _3 = w_107(D) + _2;
  wi_125 = __MEM <const double> (_3);
  if (wi_125 > 0.0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  W_126 = W_63 + wi_125;
  _4 = i_74 * xstride_109(D);
  _5 = _4 * 8ul;
  _6 = x_110(D) + _5;
  _7 = __MEM <const double> (_6);
  _8 = _7 - wm_x_67;
  _9 = wi_125 / W_126;
  _10 = _8 * _9;
  wm_x_127 = wm_x_67 + _10;
  _11 = i_74 * ystride_112(D);
  _12 = _11 * 8ul;
  _13 = y_113(D) + _12;
  _14 = __MEM <const double> (_13);
  _15 = _14 - wm_y_69;
  _16 = wi_125 / W_126;
  _17 = _15 * _16;
  wm_y_128 = wm_y_69 + _17;
  goto __BB5;

  __BB(5):
  W_62 = __PHI (__BB3: W_63, __BB4: W_126);
  wm_x_66 = __PHI (__BB3: wm_x_67, __BB4: wm_x_127);
  wm_y_68 = __PHI (__BB3: wm_y_69, __BB4: wm_y_128);
  i_129 = i_74 + 1ul;
  goto __BB6;

  __BB(6,loop_header(1)):
  W_63 = __PHI (__BB2: W_79, __BB5: W_62);
  wm_x_67 = __PHI (__BB2: wm_x_80, __BB5: wm_x_66);
  wm_y_69 = __PHI (__BB2: wm_y_81, __BB5: wm_y_68);
  i_74 = __PHI (__BB2: i_84, __BB5: i_129);
  if (i_74 < n_85(D))
    goto __BB3;
  else
    goto __BB7;

  __BB(7):
  W_86 = 0.0;
  i_87 = 0ul;
  goto __BB11;

  __BB(8):
  _18 = i_75 * wstride_106(D);
  _19 = _18 * 8ul;
  _20 = w_107(D) + _19;
  wi_118 = __MEM <const double> (_20);
  if (wi_118 > 0.0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  _21 = i_75 * xstride_109(D);
  _22 = _21 * 8ul;
  _23 = x_110(D) + _22;
  _24 = __MEM <const double> (_23);
  dx_119 = _24 - wm_x_67;
  _25 = i_75 * ystride_112(D);
  _26 = _25 * 8ul;
  _27 = y_113(D) + _26;
  _28 = __MEM <const double> (_27);
  dy_120 = _28 - wm_y_69;
  W_121 = W_65 + wi_118;
  _29 = dx_119 * dx_119;
  _30 = _29 - wm_dx2_71;
  _31 = wi_118 / W_121;
  _32 = _30 * _31;
  wm_dx2_122 = wm_dx2_71 + _32;
  _33 = dx_119 * dy_120;
  _34 = _33 - wm_dxdy_73;
  _35 = wi_118 / W_121;
  _36 = _34 * _35;
  wm_dxdy_123 = wm_dxdy_73 + _36;
  goto __BB10;

  __BB(10):
  W_64 = __PHI (__BB8: W_65, __BB9: W_121);
  wm_dx2_70 = __PHI (__BB8: wm_dx2_71, __BB9: wm_dx2_122);
  wm_dxdy_72 = __PHI (__BB8: wm_dxdy_73, __BB9: wm_dxdy_123);
  i_124 = i_75 + 1ul;
  goto __BB11;

  __BB(11,loop_header(2)):
  W_65 = __PHI (__BB7: W_86, __BB10: W_64);
  wm_dx2_71 = __PHI (__BB7: wm_dx2_82, __BB10: wm_dx2_70);
  wm_dxdy_73 = __PHI (__BB7: wm_dxdy_83, __BB10: wm_dxdy_72);
  i_75 = __PHI (__BB7: i_87, __BB10: i_124);
  if (i_75 < n_85(D))
    goto __BB8;
  else
    goto __BB12;

  __BB(12):
  d2_88 = 0.0;
  b_89 = wm_dxdy_73 / wm_dx2_71;
  _37 = wm_x_67 * b_89;
  a_90 = wm_y_69 - _37;
  __MEM <double> (c0_92(D)) = a_90;
  __MEM <double> (c1_94(D)) = b_89;
  _38 = 1.0e+0 / W_65;
  _39 = wm_x_67 * wm_x_67;
  _40 = _39 / wm_dx2_71;
  _41 = _40 + 1.0e+0;
  _42 = _38 * _41;
  __MEM <double> (cov_00_96(D)) = _42;
  _43 = W_65 * wm_dx2_71;
  _44 = 1.0e+0 / _43;
  __MEM <double> (cov_11_98(D)) = _44;
  _45 = -wm_x_67;
  _46 = W_65 * wm_dx2_71;
  _47 = _45 / _46;
  __MEM <double> (cov_01_100(D)) = _47;
  i_102 = 0ul;
  goto __BB16;

  __BB(13):
  _48 = i_76 * wstride_106(D);
  _49 = _48 * 8ul;
  _50 = w_107(D) + _49;
  wi_108 = __MEM <const double> (_50);
  if (wi_108 > 0.0)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  _51 = i_76 * xstride_109(D);
  _52 = _51 * 8ul;
  _53 = x_110(D) + _52;
  _54 = __MEM <const double> (_53);
  dx_111 = _54 - wm_x_67;
  _55 = i_76 * ystride_112(D);
  _56 = _55 * 8ul;
  _57 = y_113(D) + _56;
  _58 = __MEM <const double> (_57);
  dy_114 = _58 - wm_y_69;
  _59 = b_89 * dx_111;
  d_115 = dy_114 - _59;
  _60 = wi_108 * d_115;
  _61 = d_115 * _60;
  d2_116 = d2_78 + _61;
  goto __BB15;

  __BB(15):
  d2_77 = __PHI (__BB13: d2_78, __BB14: d2_116);
  i_117 = i_76 + 1ul;
  goto __BB16;

  __BB(16,loop_header(3)):
  i_76 = __PHI (__BB12: i_102, __BB15: i_117);
  d2_78 = __PHI (__BB12: d2_88, __BB15: d2_77);
  if (i_76 < n_85(D))
    goto __BB13;
  else
    goto __BB17;

  __BB(17):
  __MEM <double> (chisq_103(D)) = d2_78;
  _105 = 0;
  goto __BB18;

  __BB(18):
L15:
  return _105;

}


int __GIMPLE (ssa)
gsl_fit_linear_est (const double x, const double c0, const double c1, const double cov00, const double cov01, const double cov11, double * y, double * y_err)
{
  int D_4414;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  int _21;

  __BB(2):
  _1 = c1_9(D) * x_10(D);
  _2 = c0_11(D) + _1;
  __MEM <double> (y_13(D)) = _2;
  _3 = cov01_15(D) * 2.0e+0;
  _4 = cov11_16(D) * x_10(D);
  _5 = _3 + _4;
  _6 = x_10(D) * _5;
  _7 = cov00_17(D) + _6;
  _8 = sqrt (_7);
  __MEM <double> (y_err_19(D)) = _8;
  _21 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _21;

}


int __GIMPLE (ssa)
gsl_fit_mul (const double * x, const size_t xstride, const double * y, const size_t ystride, const size_t n, double * c1, double * cov_11, double * sumsq)
{
  const double d;
  const double dy;
  const double dx;
  double b;
  double d2;
  double s2;
  const double dy;
  const double dx;
  size_t i;
  double m_dxdy;
  double m_dx2;
  double m_y;
  double m_x;
  int D_4416;
  long unsigned int _1;
  long unsigned int _2;
  const double * _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  long unsigned int _9;
  long unsigned int _10;
  const double * _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  long unsigned int _17;
  long unsigned int _18;
  const double * _19;
  double _20;
  long unsigned int _21;
  long unsigned int _22;
  const double * _23;
  double _24;
  double _25;
  double _26;
  double _27;
  double _28;
  double _29;
  double _30;
  double _31;
  double _32;
  double _33;
  double _34;
  double _35;
  double _36;
  double _37;
  double _38;
  long unsigned int _39;
  long unsigned int _40;
  const double * _41;
  double _42;
  long unsigned int _43;
  long unsigned int _44;
  const double * _45;
  double _46;
  double _47;
  double _48;
  double _49;
  double _50;
  double _51;
  double _52;
  double _53;
  double _54;
  double _55;
  double _56;
  double _57;
  double _58;
  int _86;

  __BB(2):
  m_x_67 = 0.0;
  m_y_68 = 0.0;
  m_dx2_69 = 0.0;
  m_dxdy_70 = 0.0;
  i_71 = 0ul;
  goto __BB4;

  __BB(3):
  _1 = i_63 * xstride_87(D);
  _2 = _1 * 8ul;
  _3 = x_88(D) + _2;
  _4 = __MEM <const double> (_3);
  _5 = _4 - m_x_59;
  _6 = (double) i_63;
  _7 = _6 + 1.0e+0;
  _8 = _5 / _7;
  m_x_101 = m_x_59 + _8;
  _9 = i_63 * ystride_90(D);
  _10 = _9 * 8ul;
  _11 = y_91(D) + _10;
  _12 = __MEM <const double> (_11);
  _13 = _12 - m_y_60;
  _14 = (double) i_63;
  _15 = _14 + 1.0e+0;
  _16 = _13 / _15;
  m_y_102 = m_y_60 + _16;
  i_103 = i_63 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  m_x_59 = __PHI (__BB2: m_x_67, __BB3: m_x_101);
  m_y_60 = __PHI (__BB2: m_y_68, __BB3: m_y_102);
  i_63 = __PHI (__BB2: i_71, __BB3: i_103);
  if (i_63 < n_72(D))
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  i_73 = 0ul;
  goto __BB7;

  __BB(6):
  _17 = i_64 * xstride_87(D);
  _18 = _17 * 8ul;
  _19 = x_88(D) + _18;
  _20 = __MEM <const double> (_19);
  dx_96 = _20 - m_x_59;
  _21 = i_64 * ystride_90(D);
  _22 = _21 * 8ul;
  _23 = y_91(D) + _22;
  _24 = __MEM <const double> (_23);
  dy_97 = _24 - m_y_60;
  _25 = dx_96 * dx_96;
  _26 = _25 - m_dx2_61;
  _27 = (double) i_64;
  _28 = _27 + 1.0e+0;
  _29 = _26 / _28;
  m_dx2_98 = m_dx2_61 + _29;
  _30 = dx_96 * dy_97;
  _31 = _30 - m_dxdy_62;
  _32 = (double) i_64;
  _33 = _32 + 1.0e+0;
  _34 = _31 / _33;
  m_dxdy_99 = m_dxdy_62 + _34;
  i_100 = i_64 + 1ul;
  goto __BB7;

  __BB(7,loop_header(2)):
  m_dx2_61 = __PHI (__BB5: m_dx2_69, __BB6: m_dx2_98);
  m_dxdy_62 = __PHI (__BB5: m_dxdy_70, __BB6: m_dxdy_99);
  i_64 = __PHI (__BB5: i_73, __BB6: i_100);
  if (i_64 < n_72(D))
    goto __BB6;
  else
    goto __BB8;

  __BB(8):
  s2_74 = 0.0;
  d2_75 = 0.0;
  _35 = m_x_59 * m_y_60;
  _36 = m_dxdy_62 + _35;
  _37 = m_x_59 * m_x_59;
  _38 = m_dx2_61 + _37;
  b_76 = _36 / _38;
  __MEM <double> (c1_78(D)) = b_76;
  i_80 = 0ul;
  goto __BB10;

  __BB(9):
  _39 = i_65 * xstride_87(D);
  _40 = _39 * 8ul;
  _41 = x_88(D) + _40;
  _42 = __MEM <const double> (_41);
  dx_89 = _42 - m_x_59;
  _43 = i_65 * ystride_90(D);
  _44 = _43 * 8ul;
  _45 = y_91(D) + _44;
  _46 = __MEM <const double> (_45);
  dy_92 = _46 - m_y_60;
  _47 = b_76 * m_x_59;
  _48 = m_y_60 - _47;
  _49 = dy_92 + _48;
  _50 = b_76 * dx_89;
  d_93 = _49 - _50;
  _51 = d_93 * d_93;
  d2_94 = d2_66 + _51;
  i_95 = i_65 + 1ul;
  goto __BB10;

  __BB(10,loop_header(3)):
  i_65 = __PHI (__BB8: i_80, __BB9: i_95);
  d2_66 = __PHI (__BB8: d2_75, __BB9: d2_94);
  if (i_65 < n_72(D))
    goto __BB9;
  else
    goto __BB11;

  __BB(11):
  _52 = (double) n_72(D);
  _53 = _52 - 1.0e+0;
  s2_81 = d2_66 / _53;
  _54 = (double) n_72(D);
  _55 = m_x_59 * m_x_59;
  _56 = m_dx2_61 + _55;
  _57 = _54 * _56;
  _58 = s2_81 / _57;
  __MEM <double> (cov_11_82(D)) = _58;
  __MEM <double> (sumsq_84(D)) = d2_66;
  _86 = 0;
  goto __BB12;

  __BB(12):
L9:
  return _86;

}


int __GIMPLE (ssa)
gsl_fit_wmul (const double * x, const size_t xstride, const double * w, const size_t wstride, const double * y, const size_t ystride, const size_t n, double * c1, double * cov_11, double * chisq)
{
  const double d;
  const double dy;
  const double dx;
  const double wi;
  double b;
  double d2;
  const double dy;
  const double dx;
  const double wi;
  const double wi;
  size_t i;
  double wm_dxdy;
  double wm_dx2;
  double wm_y;
  double wm_x;
  double W;
  int D_4424;
  long unsigned int _1;
  long unsigned int _2;
  const double * _3;
  long unsigned int _4;
  long unsigned int _5;
  const double * _6;
  double _7;
  double _8;
  double _9;
  double _10;
  long unsigned int _11;
  long unsigned int _12;
  const double * _13;
  double _14;
  double _15;
  double _16;
  double _17;
  long unsigned int _18;
  long unsigned int _19;
  const double * _20;
  long unsigned int _21;
  long unsigned int _22;
  const double * _23;
  double _24;
  long unsigned int _25;
  long unsigned int _26;
  const double * _27;
  double _28;
  double _29;
  double _30;
  double _31;
  double _32;
  double _33;
  double _34;
  double _35;
  double _36;
  double _37;
  double _38;
  double _39;
  double _40;
  double _41;
  double _42;
  double _43;
  double _44;
  long unsigned int _45;
  long unsigned int _46;
  const double * _47;
  long unsigned int _48;
  long unsigned int _49;
  const double * _50;
  double _51;
  long unsigned int _52;
  long unsigned int _53;
  const double * _54;
  double _55;
  double _56;
  double _57;
  double _58;
  double _59;
  double _60;
  double _61;
  int _98;

  __BB(2):
  W_79 = 0.0;
  wm_x_80 = 0.0;
  wm_y_81 = 0.0;
  wm_dx2_82 = 0.0;
  wm_dxdy_83 = 0.0;
  i_84 = 0ul;
  goto __BB6;

  __BB(3):
  _1 = i_74 * wstride_99(D);
  _2 = _1 * 8ul;
  _3 = w_100(D) + _2;
  wi_118 = __MEM <const double> (_3);
  if (wi_118 > 0.0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  W_119 = W_63 + wi_118;
  _4 = i_74 * xstride_102(D);
  _5 = _4 * 8ul;
  _6 = x_103(D) + _5;
  _7 = __MEM <const double> (_6);
  _8 = _7 - wm_x_67;
  _9 = wi_118 / W_119;
  _10 = _8 * _9;
  wm_x_120 = wm_x_67 + _10;
  _11 = i_74 * ystride_105(D);
  _12 = _11 * 8ul;
  _13 = y_106(D) + _12;
  _14 = __MEM <const double> (_13);
  _15 = _14 - wm_y_69;
  _16 = wi_118 / W_119;
  _17 = _15 * _16;
  wm_y_121 = wm_y_69 + _17;
  goto __BB5;

  __BB(5):
  W_62 = __PHI (__BB3: W_63, __BB4: W_119);
  wm_x_66 = __PHI (__BB3: wm_x_67, __BB4: wm_x_120);
  wm_y_68 = __PHI (__BB3: wm_y_69, __BB4: wm_y_121);
  i_122 = i_74 + 1ul;
  goto __BB6;

  __BB(6,loop_header(1)):
  W_63 = __PHI (__BB2: W_79, __BB5: W_62);
  wm_x_67 = __PHI (__BB2: wm_x_80, __BB5: wm_x_66);
  wm_y_69 = __PHI (__BB2: wm_y_81, __BB5: wm_y_68);
  i_74 = __PHI (__BB2: i_84, __BB5: i_122);
  if (i_74 < n_85(D))
    goto __BB3;
  else
    goto __BB7;

  __BB(7):
  W_86 = 0.0;
  i_87 = 0ul;
  goto __BB11;

  __BB(8):
  _18 = i_75 * wstride_99(D);
  _19 = _18 * 8ul;
  _20 = w_100(D) + _19;
  wi_111 = __MEM <const double> (_20);
  if (wi_111 > 0.0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  _21 = i_75 * xstride_102(D);
  _22 = _21 * 8ul;
  _23 = x_103(D) + _22;
  _24 = __MEM <const double> (_23);
  dx_112 = _24 - wm_x_67;
  _25 = i_75 * ystride_105(D);
  _26 = _25 * 8ul;
  _27 = y_106(D) + _26;
  _28 = __MEM <const double> (_27);
  dy_113 = _28 - wm_y_69;
  W_114 = W_65 + wi_111;
  _29 = dx_112 * dx_112;
  _30 = _29 - wm_dx2_71;
  _31 = wi_111 / W_114;
  _32 = _30 * _31;
  wm_dx2_115 = wm_dx2_71 + _32;
  _33 = dx_112 * dy_113;
  _34 = _33 - wm_dxdy_73;
  _35 = wi_111 / W_114;
  _36 = _34 * _35;
  wm_dxdy_116 = wm_dxdy_73 + _36;
  goto __BB10;

  __BB(10):
  W_64 = __PHI (__BB8: W_65, __BB9: W_114);
  wm_dx2_70 = __PHI (__BB8: wm_dx2_71, __BB9: wm_dx2_115);
  wm_dxdy_72 = __PHI (__BB8: wm_dxdy_73, __BB9: wm_dxdy_116);
  i_117 = i_75 + 1ul;
  goto __BB11;

  __BB(11,loop_header(2)):
  W_65 = __PHI (__BB7: W_86, __BB10: W_64);
  wm_dx2_71 = __PHI (__BB7: wm_dx2_82, __BB10: wm_dx2_70);
  wm_dxdy_73 = __PHI (__BB7: wm_dxdy_83, __BB10: wm_dxdy_72);
  i_75 = __PHI (__BB7: i_87, __BB10: i_117);
  if (i_75 < n_85(D))
    goto __BB8;
  else
    goto __BB12;

  __BB(12):
  d2_88 = 0.0;
  _37 = wm_x_67 * wm_y_69;
  _38 = wm_dxdy_73 + _37;
  _39 = wm_x_67 * wm_x_67;
  _40 = wm_dx2_71 + _39;
  b_89 = _38 / _40;
  __MEM <double> (c1_91(D)) = b_89;
  _41 = wm_x_67 * wm_x_67;
  _42 = wm_dx2_71 + _41;
  _43 = W_65 * _42;
  _44 = 1.0e+0 / _43;
  __MEM <double> (cov_11_93(D)) = _44;
  i_95 = 0ul;
  goto __BB16;

  __BB(13):
  _45 = i_76 * wstride_99(D);
  _46 = _45 * 8ul;
  _47 = w_100(D) + _46;
  wi_101 = __MEM <const double> (_47);
  if (wi_101 > 0.0)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  _48 = i_76 * xstride_102(D);
  _49 = _48 * 8ul;
  _50 = x_103(D) + _49;
  _51 = __MEM <const double> (_50);
  dx_104 = _51 - wm_x_67;
  _52 = i_76 * ystride_105(D);
  _53 = _52 * 8ul;
  _54 = y_106(D) + _53;
  _55 = __MEM <const double> (_54);
  dy_107 = _55 - wm_y_69;
  _56 = b_89 * wm_x_67;
  _57 = wm_y_69 - _56;
  _58 = b_89 * dx_104;
  _59 = dy_107 - _58;
  d_108 = _57 + _59;
  _60 = wi_101 * d_108;
  _61 = d_108 * _60;
  d2_109 = d2_78 + _61;
  goto __BB15;

  __BB(15):
  d2_77 = __PHI (__BB13: d2_78, __BB14: d2_109);
  i_110 = i_76 + 1ul;
  goto __BB16;

  __BB(16,loop_header(3)):
  i_76 = __PHI (__BB12: i_95, __BB15: i_110);
  d2_78 = __PHI (__BB12: d2_88, __BB15: d2_77);
  if (i_76 < n_85(D))
    goto __BB13;
  else
    goto __BB17;

  __BB(17):
  __MEM <double> (chisq_96(D)) = d2_78;
  _98 = 0;
  goto __BB18;

  __BB(18):
L15:
  return _98;

}


int __GIMPLE (ssa)
gsl_fit_mul_est (const double x, const double c1, const double cov11, double * y, double * y_err)
{
  int D_4426;
  double _1;
  double _2;
  double _3;
  double _4;
  int _14;

  __BB(2):
  _1 = c1_5(D) * x_6(D);
  __MEM <double> (y_8(D)) = _1;
  _2 = sqrt (cov11_10(D));
  _3 = __ABS x_6(D);
  _4 = _2 * _3;
  __MEM <double> (y_err_12(D)) = _4;
  _14 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _14;

}


