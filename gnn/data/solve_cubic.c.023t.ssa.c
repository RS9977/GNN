int __GIMPLE (ssa)
gsl_poly_solve_cubic (double a, double b, double c, double * x0, double * x1, double * x2)
{
  double B;
  double A;
  double sgnR;
  double tmp;
  double tmp;
  double tmp;
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
  int D_3966;
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
  int _74;
  double _75;
  double _76;
  int _99;
  double _100;
  double _101;
  int _108;
  double _109;
  double _110;
  int _130;
  int _139;

  __BB(2):
  _1 = a_81(D) * a_81(D);
  _2 = b_82(D) * 3.0e+0;
  q_83 = _1 - _2;
  _3 = a_81(D) * 2.0e+0;
  _4 = a_81(D) * _3;
  _5 = a_81(D) * _4;
  _6 = a_81(D) * 9.0e+0;
  _7 = b_82(D) * _6;
  _8 = _5 - _7;
  _9 = c_84(D) * 2.7e+1;
  r_85 = _8 + _9;
  Q_86 = q_83 / 9.0e+0;
  R_87 = r_85 / 5.4e+1;
  _10 = Q_86 * Q_86;
  Q3_88 = Q_86 * _10;
  R2_89 = R_87 * R_87;
  _11 = r_85 * 7.29e+2;
  CR2_90 = r_85 * _11;
  _12 = q_83 * 2.916e+3;
  _13 = q_83 * _12;
  CQ3_91 = q_83 * _13;
  if (R_87 == 0.0)
    goto __BB3;
  else
    goto __BB5;

  __BB(3):
  if (Q_86 == 0.0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  _14 = -a_81(D);
  _15 = _14 / 3.0e+0;
  __MEM <double> (x0_93(D)) = _15;
  _16 = -a_81(D);
  _17 = _16 / 3.0e+0;
  __MEM <double> (x1_95(D)) = _17;
  _18 = -a_81(D);
  _19 = _18 / 3.0e+0;
  __MEM <double> (x2_97(D)) = _19;
  _99 = 3;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB24;

  __BB(5):
  if (CR2_90 == CQ3_91)
    goto __BB6;
  else
    goto __BB10;

  __BB(6):
  sqrtQ_132 = sqrt (Q_86);
  if (R_87 > 0.0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  _20 = sqrtQ_132 * -2.0e+0;
  _21 = a_81(D) / 3.0e+0;
  _22 = _20 - _21;
  __MEM <double> (x0_93(D)) = _22;
  _23 = a_81(D) / 3.0e+0;
  _24 = sqrtQ_132 - _23;
  __MEM <double> (x1_95(D)) = _24;
  _25 = a_81(D) / 3.0e+0;
  _26 = sqrtQ_132 - _25;
  __MEM <double> (x2_97(D)) = _26;
  goto __BB9;

  __BB(8):
  _27 = -sqrtQ_132;
  _28 = a_81(D) / 3.0e+0;
  _29 = _27 - _28;
  __MEM <double> (x0_93(D)) = _29;
  _30 = -sqrtQ_132;
  _31 = a_81(D) / 3.0e+0;
  _32 = _30 - _31;
  __MEM <double> (x1_95(D)) = _32;
  _33 = sqrtQ_132 * 2.0e+0;
  _34 = a_81(D) / 3.0e+0;
  _35 = _33 - _34;
  __MEM <double> (x2_97(D)) = _35;
  goto __BB9;

  __BB(9):
  _139 = 3;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB24;

  __BB(10):
  if (R2_89 < Q3_88)
    goto __BB11;
  else
    goto __BB20;

  __BB(11):
  if (R_87 >= 0.0)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  _110 = 1.0e+0;
  goto __BB14;

  __BB(13):
  _109 = -1.0e+0;
  goto __BB14;

  __BB(14):
  _75 = __PHI (__BB12: _110, __BB13: _109);
  sgnR_111 = _75;
  _36 = R2_89 / Q3_88;
  _37 = sqrt (_36);
  ratio_113 = sgnR_111 * _37;
  theta_115 = acos (ratio_113);
  _38 = sqrt (Q_86);
  norm_117 = _38 * -2.0e+0;
  _39 = theta_115 / 3.0e+0;
  _40 = cos (_39);
  _41 = norm_117 * _40;
  _42 = a_81(D) / 3.0e+0;
  _43 = _41 - _42;
  __MEM <double> (x0_93(D)) = _43;
  _44 = theta_115 + 6.28318530717958623199592693708837032318115234375e+0;
  _45 = _44 / 3.0e+0;
  _46 = cos (_45);
  _47 = norm_117 * _46;
  _48 = a_81(D) / 3.0e+0;
  _49 = _47 - _48;
  __MEM <double> (x1_95(D)) = _49;
  _50 = theta_115 - 6.28318530717958623199592693708837032318115234375e+0;
  _51 = _50 / 3.0e+0;
  _52 = cos (_51);
  _53 = norm_117 * _52;
  _54 = a_81(D) / 3.0e+0;
  _55 = _53 - _54;
  __MEM <double> (x2_97(D)) = _55;
  _56 = __MEM <double> (x0_93(D));
  _57 = __MEM <double> (x1_95(D));
  if (_56 > _57)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  tmp_121 = __MEM <double> (x1_95(D));
  _58 = __MEM <double> (x0_93(D));
  __MEM <double> (x1_95(D)) = _58;
  __MEM <double> (x0_93(D)) = tmp_121;
  goto __BB16;

  __BB(16):
  _59 = __MEM <double> (x1_95(D));
  _60 = __MEM <double> (x2_97(D));
  if (_59 > _60)
    goto __BB17;
  else
    goto __BB19;

  __BB(17):
  tmp_124 = __MEM <double> (x2_97(D));
  _61 = __MEM <double> (x1_95(D));
  __MEM <double> (x2_97(D)) = _61;
  __MEM <double> (x1_95(D)) = tmp_124;
  _62 = __MEM <double> (x0_93(D));
  _63 = __MEM <double> (x1_95(D));
  if (_62 > _63)
    goto __BB18;
  else
    goto __BB19;

  __BB(18):
  tmp_127 = __MEM <double> (x1_95(D));
  _64 = __MEM <double> (x0_93(D));
  __MEM <double> (x1_95(D)) = _64;
  __MEM <double> (x0_93(D)) = tmp_127;
  goto __BB19;

  __BB(19):
  _130 = 3;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB24;

  __BB(20):
  if (R_87 >= 0.0)
    goto __BB21;
  else
    goto __BB22;

  __BB(21):
  _101 = 1.0e+0;
  goto __BB23;

  __BB(22):
  _100 = -1.0e+0;
  goto __BB23;

  __BB(23):
  _76 = __PHI (__BB21: _101, __BB22: _100);
  sgnR_102 = _76;
  _65 = -sgnR_102;
  _66 = __ABS R_87;
  _67 = R2_89 - Q3_88;
  _68 = sqrt (_67);
  _69 = _66 + _68;
  _70 = pow (_69, 3.33333333333333314829616256247390992939472198486328125e-1);
  A_105 = _65 * _70;
  B_106 = Q_86 / A_105;
  _71 = A_105 + B_106;
  _72 = a_81(D) / 3.0e+0;
  _73 = _71 - _72;
  __MEM <double> (x0_93(D)) = _73;
  _108 = 1;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB24;

  __BB(24):
  _74 = __PHI (__BB4: _99, __BB9: _139, __BB19: _130, __BB23: _108);
L22:
  return _74;

}


