int __GIMPLE (ssa)
gsl_poly_complex_solve_cubic (double a, double b, double c, struct gsl_complex * z0, struct gsl_complex * z1, struct gsl_complex * z2)
{
  double B;
  double A;
  double sgnR;
  double tmp;
  double tmp;
  double tmp;
  double r2;
  double r1;
  double r0;
  double norm;
  double theta;
  double ratio;
  double sgnR;
  double sqrtQ;
  double CQ3;
  double CR2;
  double R2;
  double Q3;
  double R;
  double Q;
  double r;
  double q;
  double iftmp.1;
  double iftmp.0;
  int D_3969;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  double _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double _16;
  double _17;
  double _18;
  double _19;
  double _20;
  double _21;
  double _22;
  double _23;
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
  double _39;
  double _40;
  double _41;
  double _42;
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
  double _63;
  double _64;
  double _65;
  double _66;
  double _67;
  double _68;
  double _69;
  double _70;
  double _71;
  double _72;
  double _73;
  double _74;
  double _75;
  double _76;
  double _77;
  double _78;
  double _79;
  double _80;
  double _81;
  double _82;
  double _83;
  double _84;
  double _85;
  double _86;
  double _87;
  double _88;
  double _89;
  double _90;
  double _91;
  double _92;
  double _93;
  int _99;
  double _100;
  double _101;
  int _126;
  double _127;
  double _128;
  int _146;
  double _147;
  double _148;
  int _174;
  int _189;

  __BB(2):
  _1 = a_105(D) * a_105(D);
  _2 = b_106(D) * 3.0e+0;
  q_107 = _1 - _2;
  _3 = a_105(D) * 2.0e+0;
  _4 = a_105(D) * _3;
  _5 = a_105(D) * _4;
  _6 = a_105(D) * 9.0e+0;
  _7 = b_106(D) * _6;
  _8 = _5 - _7;
  _9 = c_108(D) * 2.7e+1;
  r_109 = _8 + _9;
  Q_110 = q_107 / 9.0e+0;
  R_111 = r_109 / 5.4e+1;
  _10 = Q_110 * Q_110;
  Q3_112 = Q_110 * _10;
  R2_113 = R_111 * R_111;
  _11 = r_109 * 7.29e+2;
  CR2_114 = r_109 * _11;
  _12 = q_107 * 2.916e+3;
  _13 = q_107 * _12;
  CQ3_115 = q_107 * _13;
  if (R_111 == 0.0)
    goto __BB3;
  else
    goto __BB5;

  __BB(3):
  if (Q_110 == 0.0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  _14 = -a_105(D);
  _15 = _14 / 3.0e+0;
  z0_117(D)->dat[0] = _15;
  z0_117(D)->dat[1] = 0.0;
  _16 = -a_105(D);
  _17 = _16 / 3.0e+0;
  z1_120(D)->dat[0] = _17;
  z1_120(D)->dat[1] = 0.0;
  _18 = -a_105(D);
  _19 = _18 / 3.0e+0;
  z2_123(D)->dat[0] = _19;
  z2_123(D)->dat[1] = 0.0;
  _126 = 3;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(5):
  if (CR2_114 == CQ3_115)
    goto __BB6;
  else
    goto __BB10;

  __BB(6):
  sqrtQ_176 = sqrt (Q_110);
  if (R_111 > 0.0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _20 = sqrtQ_176 * -2.0e+0;
  _21 = a_105(D) / 3.0e+0;
  _22 = _20 - _21;
  z0_117(D)->dat[0] = _22;
  z0_117(D)->dat[1] = 0.0;
  _23 = a_105(D) / 3.0e+0;
  _24 = sqrtQ_176 - _23;
  z1_120(D)->dat[0] = _24;
  z1_120(D)->dat[1] = 0.0;
  _25 = a_105(D) / 3.0e+0;
  _26 = sqrtQ_176 - _25;
  z2_123(D)->dat[0] = _26;
  z2_123(D)->dat[1] = 0.0;
  goto __BB9;

  __BB(8):
  _27 = -sqrtQ_176;
  _28 = a_105(D) / 3.0e+0;
  _29 = _27 - _28;
  z0_117(D)->dat[0] = _29;
  z0_117(D)->dat[1] = 0.0;
  _30 = -sqrtQ_176;
  _31 = a_105(D) / 3.0e+0;
  _32 = _30 - _31;
  z1_120(D)->dat[0] = _32;
  z1_120(D)->dat[1] = 0.0;
  _33 = sqrtQ_176 * 2.0e+0;
  _34 = a_105(D) / 3.0e+0;
  _35 = _33 - _34;
  z2_123(D)->dat[0] = _35;
  z2_123(D)->dat[1] = 0.0;
  goto __BB9;

  __BB(9):
  _189 = 3;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(10):
  if (R2_113 < Q3_112)
    goto __BB11;
  else
    goto __BB20;

  __BB(11):
  if (R_111 >= 0.0)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  _148 = 1.0e+0;
  goto __BB14;

  __BB(13):
  _147 = -1.0e+0;
  goto __BB14;

  __BB(14):
  _100 = __PHI (__BB12: _148, __BB13: _147);
  sgnR_149 = _100;
  _36 = R2_113 / Q3_112;
  _37 = sqrt (_36);
  ratio_151 = sgnR_149 * _37;
  theta_153 = acos (ratio_151);
  _38 = sqrt (Q_110);
  norm_155 = _38 * -2.0e+0;
  _39 = theta_153 / 3.0e+0;
  _40 = cos (_39);
  _41 = norm_155 * _40;
  _42 = a_105(D) / 3.0e+0;
  r0_156 = _41 - _42;
  _43 = theta_153 + 6.28318530717958623199592693708837032318115234375e+0;
  _44 = _43 / 3.0e+0;
  _45 = cos (_44);
  _46 = norm_155 * _45;
  _47 = a_105(D) / 3.0e+0;
  r1_157 = _46 - _47;
  _48 = theta_153 - 6.28318530717958623199592693708837032318115234375e+0;
  _49 = _48 / 3.0e+0;
  _50 = cos (_49);
  _51 = norm_155 * _50;
  _52 = a_105(D) / 3.0e+0;
  r2_158 = _51 - _52;
  if (r0_156 > r1_157)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  tmp_159 = r1_157;
  r1_160 = r0_156;
  r0_161 = tmp_159;
  goto __BB16;

  __BB(16):
  r0_94 = __PHI (__BB14: r0_156, __BB15: r0_161);
  r1_96 = __PHI (__BB14: r1_157, __BB15: r1_160);
  if (r1_96 > r2_158)
    goto __BB17;
  else
    goto __BB19;

  __BB(17):
  tmp_162 = r2_158;
  r2_163 = r1_96;
  r1_164 = tmp_162;
  if (r0_94 > r1_164)
    goto __BB18;
  else
    goto __BB19;

  __BB(18):
  tmp_165 = r1_164;
  r1_166 = r0_94;
  r0_167 = tmp_165;
  goto __BB19;

  __BB(19):
  r0_95 = __PHI (__BB16: r0_94, __BB17: r0_94, __BB18: r0_167);
  r1_97 = __PHI (__BB16: r1_96, __BB17: r1_164, __BB18: r1_166);
  r2_98 = __PHI (__BB16: r2_158, __BB17: r2_163, __BB18: r2_163);
  z0_117(D)->dat[0] = r0_95;
  z0_117(D)->dat[1] = 0.0;
  z1_120(D)->dat[0] = r1_97;
  z1_120(D)->dat[1] = 0.0;
  z2_123(D)->dat[0] = r2_98;
  z2_123(D)->dat[1] = 0.0;
  _174 = 3;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(20):
  if (R_111 >= 0.0)
    goto __BB21;
  else
    goto __BB22;

  __BB(21):
  _128 = 1.0e+0;
  goto __BB23;

  __BB(22):
  _127 = -1.0e+0;
  goto __BB23;

  __BB(23):
  _101 = __PHI (__BB21: _128, __BB22: _127);
  sgnR_129 = _101;
  _53 = -sgnR_129;
  _54 = __ABS R_111;
  _55 = R2_113 - Q3_112;
  _56 = sqrt (_55);
  _57 = _54 + _56;
  _58 = pow (_57, 3.33333333333333314829616256247390992939472198486328125e-1);
  A_132 = _53 * _58;
  B_133 = Q_110 / A_132;
  _59 = A_132 + B_133;
  if (_59 < 0.0)
    goto __BB24;
  else
    goto __BB25;

  __BB(24):
  _60 = A_132 + B_133;
  _61 = a_105(D) / 3.0e+0;
  _62 = _60 - _61;
  z0_117(D)->dat[0] = _62;
  z0_117(D)->dat[1] = 0.0;
  _63 = A_132 + B_133;
  _64 = _63 * -5.0e-1;
  _65 = a_105(D) / 3.0e+0;
  _66 = _64 - _65;
  z1_120(D)->dat[0] = _66;
  _67 = A_132 - B_133;
  _68 = __ABS _67;
  _69 = _68 * -8.660254037844385965883020617184229195117950439453125e-1;
  z1_120(D)->dat[1] = _69;
  _70 = A_132 + B_133;
  _71 = _70 * -5.0e-1;
  _72 = a_105(D) / 3.0e+0;
  _73 = _71 - _72;
  z2_123(D)->dat[0] = _73;
  _74 = A_132 - B_133;
  _75 = __ABS _74;
  _76 = _75 * 8.660254037844385965883020617184229195117950439453125e-1;
  z2_123(D)->dat[1] = _76;
  goto __BB26;

  __BB(25):
  _77 = A_132 + B_133;
  _78 = _77 * -5.0e-1;
  _79 = a_105(D) / 3.0e+0;
  _80 = _78 - _79;
  z0_117(D)->dat[0] = _80;
  _81 = A_132 - B_133;
  _82 = __ABS _81;
  _83 = _82 * -8.660254037844385965883020617184229195117950439453125e-1;
  z0_117(D)->dat[1] = _83;
  _84 = A_132 + B_133;
  _85 = _84 * -5.0e-1;
  _86 = a_105(D) / 3.0e+0;
  _87 = _85 - _86;
  z1_120(D)->dat[0] = _87;
  _88 = A_132 - B_133;
  _89 = __ABS _88;
  _90 = _89 * 8.660254037844385965883020617184229195117950439453125e-1;
  z1_120(D)->dat[1] = _90;
  _91 = A_132 + B_133;
  _92 = a_105(D) / 3.0e+0;
  _93 = _91 - _92;
  z2_123(D)->dat[0] = _93;
  z2_123(D)->dat[1] = 0.0;
  goto __BB26;

  __BB(26):
  _146 = 3;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB27;

  __BB(27):
  _99 = __PHI (__BB4: _126, __BB9: _189, __BB19: _174, __BB26: _146);
L25:
  return _99;

}


