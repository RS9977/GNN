void __GIMPLE (ssa)
set_companion_matrix (const double * a, size_t nc, double * m)
{
  size_t j;
  size_t i;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  double * _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  double * _9;
  long unsigned int _10;
  const double * _11;
  double _12;
  double _13;
  long unsigned int _14;
  const double * _15;
  double _16;
  long unsigned int _17;
  long unsigned int _18;
  long unsigned int _19;
  long unsigned int _20;
  double * _21;
  double _22;

  __BB(2):
  i_31 = 0ul;
  goto __BB7;

  __BB(3):
  j_42 = 0ul;
  goto __BB5;

  __BB(4):
  _1 = i_23 * nc_33(D);
  _2 = j_26 + _1;
  _3 = _2 * 8ul;
  _4 = m_37(D) + _3;
  __MEM <double> (_4) = 0.0;
  j_45 = j_26 + 1ul;
  goto __BB5;

  __BB(5,loop_header(4)):
  j_26 = __PHI (__BB3: j_42, __BB4: j_45);
  if (j_26 < nc_33(D))
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  i_43 = i_23 + 1ul;
  goto __BB7;

  __BB(7,loop_header(1)):
  i_23 = __PHI (__BB2: i_31, __BB6: i_43);
  if (i_23 < nc_33(D))
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  i_34 = 1ul;
  goto __BB10;

  __BB(9):
  _5 = nc_33(D) + 1ul;
  _6 = i_24 * _5;
  _7 = _6 + 18446744073709551615ul;
  _8 = _7 * 8ul;
  _9 = m_37(D) + _8;
  __MEM <double> (_9) = 1.0e+0;
  i_41 = i_24 + 1ul;
  goto __BB10;

  __BB(10,loop_header(2)):
  i_24 = __PHI (__BB8: i_34, __BB9: i_41);
  if (i_24 < nc_33(D))
    goto __BB9;
  else
    goto __BB11;

  __BB(11):
  i_35 = 0ul;
  goto __BB13;

  __BB(12):
  _10 = i_25 * 8ul;
  _11 = a_36(D) + _10;
  _12 = __MEM <const double> (_11);
  _13 = -_12;
  _14 = nc_33(D) * 8ul;
  _15 = a_36(D) + _14;
  _16 = __MEM <const double> (_15);
  _17 = i_25 + 1ul;
  _18 = nc_33(D) * _17;
  _19 = _18 + 18446744073709551615ul;
  _20 = _19 * 8ul;
  _21 = m_37(D) + _20;
  _22 = _13 / _16;
  __MEM <double> (_21) = _22;
  i_39 = i_25 + 1ul;
  goto __BB13;

  __BB(13,loop_header(3)):
  i_25 = __PHI (__BB11: i_35, __BB12: i_39);
  if (i_25 < nc_33(D))
    goto __BB12;
  else
    goto __BB14;

  __BB(14):
  return;

}


void __GIMPLE (ssa)
balance_companion_matrix (double * m, size_t nc)
{
  double s;
  double f;
  double g;
  size_t j;
  size_t i;
  double col_norm;
  double row_norm;
  int not_converged;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  double * _6;
  double _7;
  long unsigned int _8;
  long unsigned int _9;
  long unsigned int _10;
  long unsigned int _11;
  double * _12;
  double _13;
  double _14;
  long unsigned int _15;
  long unsigned int _16;
  long unsigned int _17;
  double * _18;
  double _19;
  long unsigned int _20;
  long unsigned int _21;
  long unsigned int _22;
  long unsigned int _23;
  long unsigned int _24;
  double * _25;
  double _26;
  long unsigned int _27;
  long unsigned int _28;
  long unsigned int _29;
  long unsigned int _30;
  double * _31;
  double _32;
  double _33;
  long unsigned int _34;
  long unsigned int _35;
  long unsigned int _36;
  long unsigned int _37;
  double * _38;
  double _39;
  double _40;
  double _41;
  double _42;
  double _43;
  long unsigned int _44;
  long unsigned int _45;
  double * _46;
  double _47;
  long unsigned int _48;
  long unsigned int _49;
  double * _50;
  double _51;
  long unsigned int _52;
  long unsigned int _53;
  long unsigned int _54;
  long unsigned int _55;
  double * _56;
  double _57;
  long unsigned int _58;
  long unsigned int _59;
  long unsigned int _60;
  long unsigned int _61;
  double * _62;
  double _63;
  long unsigned int _64;
  long unsigned int _65;
  long unsigned int _66;
  long unsigned int _67;
  double * _68;
  double _69;
  long unsigned int _70;
  long unsigned int _71;
  long unsigned int _72;
  long unsigned int _73;
  double * _74;
  double _75;
  long unsigned int _76;
  long unsigned int _77;
  long unsigned int _78;
  long unsigned int _79;
  double * _80;
  double _81;
  long unsigned int _82;
  long unsigned int _83;
  long unsigned int _84;
  double * _85;
  double _86;
  long unsigned int _87;
  long unsigned int _88;
  long unsigned int _89;
  long unsigned int _90;
  double * _91;
  double _92;
  long unsigned int _93;
  long unsigned int _94;
  long unsigned int _95;
  long unsigned int _96;
  double * _97;
  double _98;

  __BB(2):
  not_converged_117 = 1;
  row_norm_118 = 0.0;
  col_norm_119 = 0.0;
  goto __BB34;

  __BB(3):
  not_converged_121 = 0;
  i_122 = 0ul;
  goto __BB33;

  __BB(4):
  _1 = nc_123(D) + 18446744073709551615ul;
  if (i_107 != _1)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _2 = i_107 + 1ul;
  _3 = nc_123(D) * _2;
  _4 = i_107 + _3;
  _5 = _4 * 8ul;
  _6 = m_126(D) + _5;
  _7 = __MEM <double> (_6);
  col_norm_129 = __ABS _7;
  goto __BB9;

  __BB(6):
  col_norm_124 = 0.0;
  j_125 = 0ul;
  goto __BB8;

  __BB(7):
  _8 = j_108 + 1ul;
  _9 = nc_123(D) * _8;
  _10 = _9 + 18446744073709551615ul;
  _11 = _10 * 8ul;
  _12 = m_126(D) + _11;
  _13 = __MEM <double> (_12);
  _14 = __ABS _13;
  col_norm_127 = col_norm_103 + _14;
  j_128 = j_108 + 1ul;
  goto __BB8;

  __BB(8,loop_header(3)):
  col_norm_103 = __PHI (__BB6: col_norm_124, __BB7: col_norm_127);
  j_108 = __PHI (__BB6: j_125, __BB7: j_128);
  _15 = nc_123(D) + 18446744073709551615ul;
  if (j_108 < _15)
    goto __BB7;
  else
    goto __BB9;

  __BB(9):
  col_norm_104 = __PHI (__BB5: col_norm_129, __BB8: col_norm_103);
  if (i_107 == 0ul)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  _16 = nc_123(D) + 18446744073709551615ul;
  _17 = _16 * 8ul;
  _18 = m_126(D) + _17;
  _19 = __MEM <double> (_18);
  row_norm_132 = __ABS _19;
  goto __BB14;

  __BB(11):
  _20 = nc_123(D) + 18446744073709551615ul;
  if (i_107 == _20)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  _21 = nc_123(D) + 1ul;
  _22 = i_107 * _21;
  _23 = _22 + 18446744073709551615ul;
  _24 = _23 * 8ul;
  _25 = m_126(D) + _24;
  _26 = __MEM <double> (_25);
  row_norm_131 = __ABS _26;
  goto __BB14;

  __BB(13):
  _27 = nc_123(D) + 1ul;
  _28 = i_107 * _27;
  _29 = _28 + 18446744073709551615ul;
  _30 = _29 * 8ul;
  _31 = m_126(D) + _30;
  _32 = __MEM <double> (_31);
  _33 = __ABS _32;
  _34 = i_107 + 1ul;
  _35 = nc_123(D) * _34;
  _36 = _35 + 18446744073709551615ul;
  _37 = _36 * 8ul;
  _38 = m_126(D) + _37;
  _39 = __MEM <double> (_38);
  _40 = __ABS _39;
  row_norm_130 = _33 + _40;
  goto __BB14;

  __BB(14):
  row_norm_102 = __PHI (__BB10: row_norm_132, __BB12: row_norm_131, __BB13: row_norm_130);
  if (col_norm_104 == 0.0)
    goto __BB16;
  else
    goto __BB15;

  __BB(15):
  if (row_norm_102 == 0.0)
    goto __BB16;
  else
    goto __BB17;

  __BB(16):
  // predicted unlikely by continue predictor.
  goto __BB32;

  __BB(17):
  g_133 = row_norm_102 / 2.0e+0;
  f_134 = 1.0e+0;
  s_135 = col_norm_104 + row_norm_102;
  goto __BB19;

  __BB(18):
  f_148 = f_110 * 2.0e+0;
  col_norm_149 = col_norm_105 * 4.0e+0;
  goto __BB19;

  __BB(19,loop_header(4)):
  col_norm_105 = __PHI (__BB17: col_norm_104, __BB18: col_norm_149);
  f_110 = __PHI (__BB17: f_134, __BB18: f_148);
  if (col_norm_105 < g_133)
    goto __BB18;
  else
    goto __BB20;

  __BB(20):
  g_136 = row_norm_102 * 2.0e+0;
  goto __BB22;

  __BB(21):
  f_146 = f_111 / 2.0e+0;
  col_norm_147 = col_norm_106 / 4.0e+0;
  goto __BB22;

  __BB(22,loop_header(5)):
  col_norm_106 = __PHI (__BB20: col_norm_105, __BB21: col_norm_147);
  f_111 = __PHI (__BB20: f_110, __BB21: f_146);
  if (col_norm_106 > g_136)
    goto __BB21;
  else
    goto __BB23;

  __BB(23):
  _41 = row_norm_102 + col_norm_106;
  _42 = s_135 * 9.499999999999999555910790149937383830547332763671875e-1;
  _43 = f_111 * _42;
  if (_41 < _43)
    goto __BB24;
  else
    goto __BB32;

  __BB(24):
  not_converged_137 = 1;
  g_138 = 1.0e+0 / f_111;
  if (i_107 == 0ul)
    goto __BB25;
  else
    goto __BB26;

  __BB(25):
  _44 = nc_123(D) + 18446744073709551615ul;
  _45 = _44 * 8ul;
  _46 = m_126(D) + _45;
  _47 = __MEM <double> (_46);
  _48 = nc_123(D) + 18446744073709551615ul;
  _49 = _48 * 8ul;
  _50 = m_126(D) + _49;
  _51 = g_138 * _47;
  __MEM <double> (_50) = _51;
  goto __BB27;

  __BB(26):
  _52 = nc_123(D) + 1ul;
  _53 = i_107 * _52;
  _54 = _53 + 18446744073709551615ul;
  _55 = _54 * 8ul;
  _56 = m_126(D) + _55;
  _57 = __MEM <double> (_56);
  _58 = nc_123(D) + 1ul;
  _59 = i_107 * _58;
  _60 = _59 + 18446744073709551615ul;
  _61 = _60 * 8ul;
  _62 = m_126(D) + _61;
  _63 = g_138 * _57;
  __MEM <double> (_62) = _63;
  _64 = i_107 + 1ul;
  _65 = nc_123(D) * _64;
  _66 = _65 + 18446744073709551615ul;
  _67 = _66 * 8ul;
  _68 = m_126(D) + _67;
  _69 = __MEM <double> (_68);
  _70 = i_107 + 1ul;
  _71 = nc_123(D) * _70;
  _72 = _71 + 18446744073709551615ul;
  _73 = _72 * 8ul;
  _74 = m_126(D) + _73;
  _75 = g_138 * _69;
  __MEM <double> (_74) = _75;
  goto __BB27;

  __BB(27):
  _76 = nc_123(D) + 18446744073709551615ul;
  if (i_107 == _76)
    goto __BB28;
  else
    goto __BB31;

  __BB(28):
  j_143 = 0ul;
  goto __BB30;

  __BB(29):
  _77 = j_109 * nc_123(D);
  _78 = i_107 + _77;
  _79 = _78 * 8ul;
  _80 = m_126(D) + _79;
  _81 = __MEM <double> (_80);
  _82 = j_109 * nc_123(D);
  _83 = i_107 + _82;
  _84 = _83 * 8ul;
  _85 = m_126(D) + _84;
  _86 = f_111 * _81;
  __MEM <double> (_85) = _86;
  j_145 = j_109 + 1ul;
  goto __BB30;

  __BB(30,loop_header(6)):
  j_109 = __PHI (__BB28: j_143, __BB29: j_145);
  if (j_109 < nc_123(D))
    goto __BB29;
  else
    goto __BB32;

  __BB(31):
  _87 = i_107 + 1ul;
  _88 = nc_123(D) * _87;
  _89 = i_107 + _88;
  _90 = _89 * 8ul;
  _91 = m_126(D) + _90;
  _92 = __MEM <double> (_91);
  _93 = i_107 + 1ul;
  _94 = nc_123(D) * _93;
  _95 = i_107 + _94;
  _96 = _95 * 8ul;
  _97 = m_126(D) + _96;
  _98 = f_111 * _92;
  __MEM <double> (_97) = _98;
  goto __BB32;

  __BB(32):
  not_converged_99 = __PHI (__BB16: not_converged_100, __BB23: not_converged_100, __BB30: not_converged_137, __BB31: not_converged_137);
  i_150 = i_107 + 1ul;
  goto __BB33;

  __BB(33,loop_header(2)):
  not_converged_100 = __PHI (__BB3: not_converged_121, __BB32: not_converged_99);
  i_107 = __PHI (__BB3: i_122, __BB32: i_150);
  if (i_107 < nc_123(D))
    goto __BB4;
  else
    goto __BB34;

  __BB(34,loop_header(1)):
  not_converged_101 = __PHI (__BB2: not_converged_117, __BB33: not_converged_100);
  if (not_converged_101 != 0)
    goto __BB3;
  else
    goto __BB35;

  __BB(35):
  return;

}


int __GIMPLE (ssa)
qr_companion (double * h, size_t nc, double * zroot)
{
  double a3;
  double a2;
  double a1;
  double a3;
  double a2;
  double a1;
  size_t n;
  int notlast;
  double r;
  double q;
  double p;
  double z;
  double s;
  double y;
  double x;
  double w;
  size_t m;
  size_t k;
  size_t j;
  size_t i;
  size_t e;
  size_t iterations;
  double t;
  double iftmp.0;
  int D_4511;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;
  double * _6;
  double _7;
  long unsigned int _8;
  long unsigned int _9;
  long unsigned int _10;
  long unsigned int _11;
  double * _12;
  double _13;
  long unsigned int _14;
  long unsigned int _15;
  long unsigned int _16;
  long unsigned int _17;
  double * _18;
  double _19;
  double _20;
  double _21;
  long unsigned int _22;
  long unsigned int _23;
  long unsigned int _24;
  long unsigned int _25;
  double * _26;
  long unsigned int _27;
  long unsigned int _28;
  double * _29;
  double _30;
  long unsigned int _31;
  long unsigned int _32;
  sizetype _33;
  double * _34;
  long unsigned int _35;
  long unsigned int _36;
  long unsigned int _37;
  long unsigned int _38;
  double * _39;
  long unsigned int _40;
  long unsigned int _41;
  long unsigned int _42;
  long unsigned int _43;
  long unsigned int _44;
  double * _45;
  double _46;
  long unsigned int _47;
  long unsigned int _48;
  long unsigned int _49;
  long unsigned int _50;
  long unsigned int _51;
  double * _52;
  double _53;
  long unsigned int _54;
  double _55;
  double _56;
  double _57;
  double _58;
  long unsigned int _59;
  long unsigned int _60;
  double * _61;
  double _62;
  long unsigned int _63;
  long unsigned int _64;
  sizetype _65;
  double * _66;
  long unsigned int _67;
  long unsigned int _68;
  double * _69;
  double _70;
  long unsigned int _71;
  long unsigned int _72;
  sizetype _73;
  double * _74;
  long unsigned int _75;
  long unsigned int _76;
  double * _77;
  double _78;
  long unsigned int _79;
  long unsigned int _80;
  sizetype _81;
  double * _82;
  double _83;
  long unsigned int _84;
  long unsigned int _85;
  double * _86;
  double _87;
  long unsigned int _88;
  long unsigned int _89;
  sizetype _90;
  double * _91;
  long unsigned int _92;
  long unsigned int _93;
  long unsigned int _94;
  long unsigned int _95;
  long unsigned int _96;
  double * _97;
  double _98;
  long unsigned int _99;
  long unsigned int _100;
  long unsigned int _101;
  long unsigned int _102;
  double * _103;
  double _104;
  long unsigned int _105;
  long unsigned int _106;
  long unsigned int _107;
  long unsigned int _108;
  long unsigned int _109;
  double * _110;
  double _111;
  double _112;
  long unsigned int _113;
  long unsigned int _114;
  long unsigned int _115;
  long unsigned int _116;
  long unsigned int _117;
  double * _118;
  double _119;
  double _120;
  double _121;
  long unsigned int _122;
  long unsigned int _123;
  long unsigned int _124;
  long unsigned int _125;
  double * _126;
  long unsigned int _127;
  long unsigned int _128;
  long unsigned int _129;
  long unsigned int _130;
  double * _131;
  double _132;
  double _133;
  double _134;
  long unsigned int _135;
  long unsigned int _136;
  long unsigned int _137;
  long unsigned int _138;
  double * _139;
  double _140;
  double _141;
  long unsigned int _142;
  long unsigned int _143;
  long unsigned int _144;
  double * _145;
  double _146;
  double _147;
  double _148;
  long unsigned int _149;
  long unsigned int _150;
  long unsigned int _151;
  long unsigned int _152;
  double * _153;
  double _154;
  double _155;
  double _156;
  double _157;
  long unsigned int _158;
  long unsigned int _159;
  long unsigned int _160;
  long unsigned int _161;
  long unsigned int _162;
  double * _163;
  double _164;
  long unsigned int _165;
  long unsigned int _166;
  long unsigned int _167;
  long unsigned int _168;
  double * _169;
  double _170;
  long unsigned int _171;
  long unsigned int _172;
  long unsigned int _173;
  double * _174;
  double _175;
  double _176;
  double _177;
  double _178;
  double _179;
  double _180;
  double _181;
  double _182;
  double _183;
  long unsigned int _184;
  long unsigned int _185;
  long unsigned int _186;
  long unsigned int _187;
  long unsigned int _188;
  double * _189;
  long unsigned int _190;
  long unsigned int _191;
  long unsigned int _192;
  long unsigned int _193;
  long unsigned int _194;
  double * _195;
  long unsigned int _196;
  _Bool _197;
  long unsigned int _198;
  long unsigned int _199;
  long unsigned int _200;
  long unsigned int _201;
  long unsigned int _202;
  double * _203;
  long unsigned int _204;
  long unsigned int _205;
  long unsigned int _206;
  long unsigned int _207;
  double * _208;
  long unsigned int _209;
  long unsigned int _210;
  long unsigned int _211;
  long unsigned int _212;
  long unsigned int _213;
  double * _214;
  double _215;
  double _216;
  double _217;
  double _218;
  double _219;
  double _220;
  double _221;
  double _222;
  double _223;
  double _224;
  long unsigned int _225;
  long unsigned int _226;
  long unsigned int _227;
  long unsigned int _228;
  long unsigned int _229;
  double * _230;
  double _231;
  long unsigned int _232;
  long unsigned int _233;
  long unsigned int _234;
  long unsigned int _235;
  long unsigned int _236;
  double * _237;
  double _238;
  long unsigned int _239;
  long unsigned int _240;
  long unsigned int _241;
  long unsigned int _242;
  long unsigned int _243;
  double * _244;
  double _245;
  long unsigned int _246;
  long unsigned int _247;
  long unsigned int _248;
  long unsigned int _249;
  long unsigned int _250;
  double * _251;
  double _252;
  long unsigned int _253;
  long unsigned int _254;
  long unsigned int _255;
  long unsigned int _256;
  double * _257;
  double _258;
  double _259;
  long unsigned int _260;
  long unsigned int _261;
  long unsigned int _262;
  long unsigned int _263;
  long unsigned int _264;
  double * _265;
  double _266;
  double _267;
  long unsigned int _268;
  long unsigned int _269;
  long unsigned int _270;
  long unsigned int _271;
  long unsigned int _272;
  double * _273;
  double _274;
  double _275;
  long unsigned int _276;
  long unsigned int _277;
  long unsigned int _278;
  long unsigned int _279;
  long unsigned int _280;
  double * _281;
  double _282;
  long unsigned int _283;
  long unsigned int _284;
  long unsigned int _285;
  long unsigned int _286;
  double * _287;
  double _288;
  double _289;
  long unsigned int _290;
  long unsigned int _291;
  long unsigned int _292;
  long unsigned int _293;
  double * _294;
  double _295;
  long unsigned int _296;
  long unsigned int _297;
  long unsigned int _298;
  long unsigned int _299;
  long unsigned int _300;
  double * _301;
  double _302;
  double _303;
  long unsigned int _304;
  long unsigned int _305;
  long unsigned int _306;
  long unsigned int _307;
  long unsigned int _308;
  double * _309;
  double _310;
  long unsigned int _311;
  long unsigned int _312;
  long unsigned int _313;
  long unsigned int _314;
  long unsigned int _315;
  long unsigned int _316;
  double * _317;
  double _318;
  double _319;
  long unsigned int _320;
  long unsigned int _321;
  long unsigned int _322;
  long unsigned int _323;
  double * _324;
  double _325;
  double _326;
  long unsigned int _327;
  long unsigned int _328;
  long unsigned int _329;
  long unsigned int _330;
  long unsigned int _331;
  double * _332;
  double _333;
  double _334;
  long unsigned int _335;
  long unsigned int _336;
  long unsigned int _337;
  long unsigned int _338;
  long unsigned int _339;
  double * _340;
  double _341;
  double _342;
  long unsigned int _343;
  long unsigned int _344;
  long unsigned int _345;
  long unsigned int _346;
  long unsigned int _347;
  double * _348;
  double _349;
  long unsigned int _350;
  long unsigned int _351;
  long unsigned int _352;
  long unsigned int _353;
  double * _354;
  double _355;
  double _356;
  long unsigned int _357;
  long unsigned int _358;
  long unsigned int _359;
  long unsigned int _360;
  double * _361;
  double _362;
  long unsigned int _363;
  long unsigned int _364;
  long unsigned int _365;
  long unsigned int _366;
  long unsigned int _367;
  double * _368;
  double _369;
  long unsigned int _370;
  long unsigned int _371;
  long unsigned int _372;
  long unsigned int _373;
  long unsigned int _374;
  double * _375;
  double _376;
  long unsigned int _377;
  int _424;
  double _425;
  double _488;
  double _489;
  int _526;
  int _547;

  __BB(2):
  t_441 = 0.0;
  p_442 = 0.0;
  q_443 = 0.0;
  r_444 = 0.0;
  n_446 = nc_445(D);
  goto __BB3;

  __BB(3,loop_header(1)):
  t_378 = __PHI (__BB2: t_441, __BB12: t_379, __BB19: t_379);
  p_398 = __PHI (__BB2: p_442, __BB12: p_399, __BB19: p_527);
  q_409 = __PHI (__BB2: q_443, __BB12: q_410, __BB19: q_528);
  r_416 = __PHI (__BB2: r_444, __BB12: r_417, __BB19: r_417);
  n_423 = __PHI (__BB2: n_446, __BB12: n_546, __BB19: n_543);
next_root:
  if (n_423 == 0ul)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  _547 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB68;

  __BB(5):
  iterations_448 = 0ul;
  goto __BB6;

  __BB(6,loop_header(2)):
  t_379 = __PHI (__BB5: t_378, __BB67: t_380);
  iterations_381 = __PHI (__BB5: iterations_448, __BB67: iterations_466);
  p_399 = __PHI (__BB5: p_398, __BB67: p_408);
  q_410 = __PHI (__BB5: q_409, __BB67: q_415);
  r_417 = __PHI (__BB5: r_416, __BB67: r_422);
next_iteration:
  e_449 = n_423;
  goto __BB10;

  __BB(7):
  _1 = e_382 + 18446744073709551615ul;
  _2 = nc_445(D) * _1;
  _3 = e_382 + _2;
  _4 = _3 + 18446744073709551614ul;
  _5 = _4 * 8ul;
  _6 = h_450(D) + _5;
  _7 = __MEM <double> (_6);
  a1_451 = __ABS _7;
  _8 = nc_445(D) + 1ul;
  _9 = e_382 + 18446744073709551614ul;
  _10 = _8 * _9;
  _11 = _10 * 8ul;
  _12 = h_450(D) + _11;
  _13 = __MEM <double> (_12);
  a2_452 = __ABS _13;
  _14 = nc_445(D) + 1ul;
  _15 = e_382 + 18446744073709551615ul;
  _16 = _14 * _15;
  _17 = _16 * 8ul;
  _18 = h_450(D) + _17;
  _19 = __MEM <double> (_18);
  a3_453 = __ABS _19;
  _20 = a2_452 + a3_453;
  _21 = _20 * 2.220446049250313080847263336181640625e-16;
  if (a1_451 <= _21)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  goto __BB11;

  __BB(9):
  e_454 = e_382 + 18446744073709551615ul;
  goto __BB10;

  __BB(10,loop_header(3)):
  e_382 = __PHI (__BB6: e_449, __BB9: e_454);
  if (e_382 > 1ul)
    goto __BB7;
  else
    goto __BB11;

  __BB(11):
  _22 = nc_445(D) + 1ul;
  _23 = n_423 + 18446744073709551615ul;
  _24 = _22 * _23;
  _25 = _24 * 8ul;
  _26 = h_450(D) + _25;
  x_455 = __MEM <double> (_26);
  if (e_382 == n_423)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  _27 = n_423 + 18446744073709551615ul;
  _28 = _27 * 16ul;
  _29 = zroot_532(D) + _28;
  _30 = x_455 + t_379;
  __MEM <double> ((double *)_29) = _30;
  _31 = n_423 + 18446744073709551615ul;
  _32 = _31 * 16ul;
  _33 = _32 + 8ul;
  _34 = zroot_532(D) + _33;
  __MEM <double> ((double *)_34) = 0.0;
  n_546 = n_423 + 18446744073709551615ul;
  // predicted unlikely by goto predictor.
  goto __BB3;

  __BB(13):
  _35 = nc_445(D) + 1ul;
  _36 = n_423 + 18446744073709551614ul;
  _37 = _35 * _36;
  _38 = _37 * 8ul;
  _39 = h_450(D) + _38;
  y_456 = __MEM <double> (_39);
  _40 = n_423 + 18446744073709551614ul;
  _41 = nc_445(D) * _40;
  _42 = n_423 + _41;
  _43 = _42 + 18446744073709551615ul;
  _44 = _43 * 8ul;
  _45 = h_450(D) + _44;
  _46 = __MEM <double> (_45);
  _47 = n_423 + 18446744073709551615ul;
  _48 = nc_445(D) * _47;
  _49 = n_423 + _48;
  _50 = _49 + 18446744073709551614ul;
  _51 = _50 * 8ul;
  _52 = h_450(D) + _51;
  _53 = __MEM <double> (_52);
  w_457 = _46 * _53;
  _54 = n_423 + 18446744073709551615ul;
  if (e_382 == _54)
    goto __BB14;
  else
    goto __BB20;

  __BB(14):
  _55 = y_456 - x_455;
  p_527 = _55 / 2.0e+0;
  _56 = p_527 * p_527;
  q_528 = w_457 + _56;
  _57 = __ABS q_528;
  y_530 = sqrt (_57);
  x_531 = x_455 + t_379;
  if (q_528 > 0.0)
    goto __BB15;
  else
    goto __BB18;

  __BB(15):
  if (p_527 < 0.0)
    goto __BB16;
  else
    goto __BB17;

  __BB(16):
  y_537 = -y_530;
  goto __BB17;

  __BB(17):
  y_395 = __PHI (__BB15: y_530, __BB16: y_537);
  y_538 = y_395 + p_527;
  _58 = w_457 / y_538;
  _59 = n_423 + 18446744073709551615ul;
  _60 = _59 * 16ul;
  _61 = zroot_532(D) + _60;
  _62 = x_531 - _58;
  __MEM <double> ((double *)_61) = _62;
  _63 = n_423 + 18446744073709551615ul;
  _64 = _63 * 16ul;
  _65 = _64 + 8ul;
  _66 = zroot_532(D) + _65;
  __MEM <double> ((double *)_66) = 0.0;
  _67 = n_423 + 18446744073709551614ul;
  _68 = _67 * 16ul;
  _69 = zroot_532(D) + _68;
  _70 = x_531 + y_538;
  __MEM <double> ((double *)_69) = _70;
  _71 = n_423 + 18446744073709551614ul;
  _72 = _71 * 16ul;
  _73 = _72 + 8ul;
  _74 = zroot_532(D) + _73;
  __MEM <double> ((double *)_74) = 0.0;
  goto __BB19;

  __BB(18):
  _75 = n_423 + 18446744073709551615ul;
  _76 = _75 * 16ul;
  _77 = zroot_532(D) + _76;
  _78 = x_531 + p_527;
  __MEM <double> ((double *)_77) = _78;
  _79 = n_423 + 18446744073709551615ul;
  _80 = _79 * 16ul;
  _81 = _80 + 8ul;
  _82 = zroot_532(D) + _81;
  _83 = -y_530;
  __MEM <double> ((double *)_82) = _83;
  _84 = n_423 + 18446744073709551614ul;
  _85 = _84 * 16ul;
  _86 = zroot_532(D) + _85;
  _87 = x_531 + p_527;
  __MEM <double> ((double *)_86) = _87;
  _88 = n_423 + 18446744073709551614ul;
  _89 = _88 * 16ul;
  _90 = _89 + 8ul;
  _91 = zroot_532(D) + _90;
  __MEM <double> ((double *)_91) = y_530;
  goto __BB19;

  __BB(19):
  n_543 = n_423 + 18446744073709551614ul;
  // predicted unlikely by goto predictor.
  goto __BB3;

  __BB(20):
  if (iterations_381 == 120ul)
    goto __BB21;
  else
    goto __BB22;

  __BB(21):
  _526 = _Literal (int) -1;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB68;

  __BB(22):
  _92 = iterations_381 % 10ul;
  if (_92 == 0ul)
    goto __BB23;
  else
    goto __BB28;

  __BB(23):
  if (iterations_381 != 0ul)
    goto __BB24;
  else
    goto __BB28;

  __BB(24):
  t_458 = t_379 + x_455;
  i_459 = 1ul;
  goto __BB26;

  __BB(25):
  _93 = nc_445(D) + 1ul;
  _94 = i_383 + 18446744073709551615ul;
  _95 = _93 * _94;
  _96 = _95 * 8ul;
  _97 = h_450(D) + _96;
  _98 = __MEM <double> (_97);
  _99 = nc_445(D) + 1ul;
  _100 = i_383 + 18446744073709551615ul;
  _101 = _99 * _100;
  _102 = _101 * 8ul;
  _103 = h_450(D) + _102;
  _104 = _98 - x_455;
  __MEM <double> (_103) = _104;
  i_465 = i_383 + 1ul;
  goto __BB26;

  __BB(26,loop_header(4)):
  i_383 = __PHI (__BB24: i_459, __BB25: i_465);
  if (i_383 <= n_423)
    goto __BB25;
  else
    goto __BB27;

  __BB(27):
  _105 = n_423 + 18446744073709551615ul;
  _106 = nc_445(D) * _105;
  _107 = n_423 + _106;
  _108 = _107 + 18446744073709551614ul;
  _109 = _108 * 8ul;
  _110 = h_450(D) + _109;
  _111 = __MEM <double> (_110);
  _112 = __ABS _111;
  _113 = n_423 + 18446744073709551614ul;
  _114 = nc_445(D) * _113;
  _115 = n_423 + _114;
  _116 = _115 + 18446744073709551613ul;
  _117 = _116 * 8ul;
  _118 = h_450(D) + _117;
  _119 = __MEM <double> (_118);
  _120 = __ABS _119;
  s_460 = _112 + _120;
  y_461 = s_460 * 7.5e-1;
  x_462 = y_461;
  _121 = s_460 * -4.375e-1;
  w_463 = s_460 * _121;
  goto __BB28;

  __BB(28):
  t_380 = __PHI (__BB22: t_379, __BB23: t_379, __BB27: t_458);
  w_390 = __PHI (__BB22: w_457, __BB23: w_457, __BB27: w_463);
  x_391 = __PHI (__BB22: x_455, __BB23: x_455, __BB27: x_462);
  y_396 = __PHI (__BB22: y_456, __BB23: y_456, __BB27: y_461);
  iterations_466 = iterations_381 + 1ul;
  m_467 = n_423 + 18446744073709551614ul;
  goto __BB34;

  __BB(29):
  _122 = nc_445(D) + 1ul;
  _123 = m_389 + 18446744073709551615ul;
  _124 = _122 * _123;
  _125 = _124 * 8ul;
  _126 = h_450(D) + _125;
  z_468 = __MEM <double> (_126);
  r_469 = x_391 - z_468;
  s_470 = y_396 - z_468;
  _127 = m_389 + 18446744073709551615ul;
  _128 = nc_445(D) * _127;
  _129 = m_389 + _128;
  _130 = _129 * 8ul;
  _131 = h_450(D) + _130;
  _132 = __MEM <double> (_131);
  _133 = r_469 * s_470;
  _134 = _133 - w_390;
  _135 = nc_445(D) + 1ul;
  _136 = m_389 * _135;
  _137 = _136 + 18446744073709551615ul;
  _138 = _137 * 8ul;
  _139 = h_450(D) + _138;
  _140 = __MEM <double> (_139);
  _141 = _134 / _140;
  p_471 = _132 + _141;
  _142 = nc_445(D) + 1ul;
  _143 = m_389 * _142;
  _144 = _143 * 8ul;
  _145 = h_450(D) + _144;
  _146 = __MEM <double> (_145);
  _147 = _146 - z_468;
  _148 = _147 - r_469;
  q_472 = _148 - s_470;
  _149 = m_389 + 1ul;
  _150 = nc_445(D) * _149;
  _151 = m_389 + _150;
  _152 = _151 * 8ul;
  _153 = h_450(D) + _152;
  r_473 = __MEM <double> (_153);
  _154 = __ABS p_471;
  _155 = __ABS q_472;
  _156 = _154 + _155;
  _157 = __ABS r_473;
  s_474 = _156 + _157;
  p_475 = p_471 / s_474;
  q_476 = q_472 / s_474;
  r_477 = r_473 / s_474;
  if (m_389 == e_382)
    goto __BB30;
  else
    goto __BB31;

  __BB(30):
  goto __BB35;

  __BB(31):
  _158 = m_389 + 18446744073709551615ul;
  _159 = nc_445(D) * _158;
  _160 = m_389 + _159;
  _161 = _160 + 18446744073709551614ul;
  _162 = _161 * 8ul;
  _163 = h_450(D) + _162;
  _164 = __MEM <double> (_163);
  a1_478 = __ABS _164;
  _165 = nc_445(D) + 1ul;
  _166 = m_389 + 18446744073709551614ul;
  _167 = _165 * _166;
  _168 = _167 * 8ul;
  _169 = h_450(D) + _168;
  _170 = __MEM <double> (_169);
  a2_479 = __ABS _170;
  _171 = nc_445(D) + 1ul;
  _172 = m_389 * _171;
  _173 = _172 * 8ul;
  _174 = h_450(D) + _173;
  _175 = __MEM <double> (_174);
  a3_480 = __ABS _175;
  _176 = __ABS q_476;
  _177 = __ABS r_477;
  _178 = _176 + _177;
  _179 = a1_478 * _178;
  _180 = __ABS p_475;
  _181 = _180 * 2.220446049250313080847263336181640625e-16;
  _182 = a2_479 + a3_480;
  _183 = _181 * _182;
  if (_179 <= _183)
    goto __BB32;
  else
    goto __BB33;

  __BB(32):
  goto __BB35;

  __BB(33):
  m_481 = m_389 + 18446744073709551615ul;
  goto __BB34;

  __BB(34,loop_header(5)):
  m_389 = __PHI (__BB28: m_467, __BB33: m_481);
  p_400 = __PHI (__BB28: p_399, __BB33: p_475);
  q_411 = __PHI (__BB28: q_410, __BB33: q_476);
  r_418 = __PHI (__BB28: r_417, __BB33: r_477);
  if (m_389 >= e_382)
    goto __BB29;
  else
    goto __BB35;

  __BB(35):
  p_401 = __PHI (__BB30: p_475, __BB32: p_475, __BB34: p_400);
  q_412 = __PHI (__BB30: q_476, __BB32: q_476, __BB34: q_411);
  r_419 = __PHI (__BB30: r_477, __BB32: r_477, __BB34: r_418);
  i_482 = m_389 + 2ul;
  goto __BB37;

  __BB(36):
  _184 = i_384 + 18446744073709551615ul;
  _185 = nc_445(D) * _184;
  _186 = i_384 + _185;
  _187 = _186 + 18446744073709551613ul;
  _188 = _187 * 8ul;
  _189 = h_450(D) + _188;
  __MEM <double> (_189) = 0.0;
  i_525 = i_384 + 1ul;
  goto __BB37;

  __BB(37,loop_header(6)):
  i_384 = __PHI (__BB35: i_482, __BB36: i_525);
  if (i_384 <= n_423)
    goto __BB36;
  else
    goto __BB38;

  __BB(38):
  i_483 = m_389 + 3ul;
  goto __BB40;

  __BB(39):
  _190 = i_385 + 18446744073709551615ul;
  _191 = nc_445(D) * _190;
  _192 = i_385 + _191;
  _193 = _192 + 18446744073709551612ul;
  _194 = _193 * 8ul;
  _195 = h_450(D) + _194;
  __MEM <double> (_195) = 0.0;
  i_523 = i_385 + 1ul;
  goto __BB40;

  __BB(40,loop_header(7)):
  i_385 = __PHI (__BB38: i_483, __BB39: i_523);
  if (i_385 <= n_423)
    goto __BB39;
  else
    goto __BB41;

  __BB(41):
  k_484 = m_389;
  goto __BB66;

  __BB(42):
  _196 = n_423 + 18446744073709551615ul;
  _197 = k_388 != _196;
  notlast_485 = (int) _197;
  if (k_388 != m_389)
    goto __BB43;
  else
    goto __BB49;

  __BB(43):
  _198 = k_388 + 18446744073709551615ul;
  _199 = nc_445(D) * _198;
  _200 = k_388 + _199;
  _201 = _200 + 18446744073709551614ul;
  _202 = _201 * 8ul;
  _203 = h_450(D) + _202;
  p_486 = __MEM <double> (_203);
  _204 = nc_445(D) + 1ul;
  _205 = k_388 * _204;
  _206 = _205 + 18446744073709551614ul;
  _207 = _206 * 8ul;
  _208 = h_450(D) + _207;
  q_487 = __MEM <double> (_208);
  if (notlast_485 != 0)
    goto __BB44;
  else
    goto __BB45;

  __BB(44):
  _209 = k_388 + 1ul;
  _210 = nc_445(D) * _209;
  _211 = k_388 + _210;
  _212 = _211 + 18446744073709551614ul;
  _213 = _212 * 8ul;
  _214 = h_450(D) + _213;
  _489 = __MEM <double> (_214);
  goto __BB46;

  __BB(45):
  _488 = 0.0;
  goto __BB46;

  __BB(46):
  _425 = __PHI (__BB44: _489, __BB45: _488);
  r_490 = _425;
  _215 = __ABS p_486;
  _216 = __ABS q_487;
  _217 = _215 + _216;
  _218 = __ABS r_490;
  x_491 = _217 + _218;
  if (x_491 == 0.0)
    goto __BB47;
  else
    goto __BB48;

  __BB(47):
  // predicted unlikely by continue predictor.
  goto __BB65;

  __BB(48):
  p_492 = p_486 / x_491;
  q_493 = q_487 / x_491;
  r_494 = r_490 / x_491;
  goto __BB49;

  __BB(49):
  x_392 = __PHI (__BB42: x_394, __BB48: x_491);
  p_402 = __PHI (__BB42: p_408, __BB48: p_492);
  q_413 = __PHI (__BB42: q_415, __BB48: q_493);
  r_420 = __PHI (__BB42: r_422, __BB48: r_494);
  _219 = p_402 * p_402;
  _220 = q_413 * q_413;
  _221 = _219 + _220;
  _222 = r_420 * r_420;
  _223 = _221 + _222;
  s_496 = sqrt (_223);
  if (p_402 < 0.0)
    goto __BB50;
  else
    goto __BB51;

  __BB(50):
  s_497 = -s_496;
  goto __BB51;

  __BB(51):
  s_397 = __PHI (__BB49: s_496, __BB50: s_497);
  if (k_388 != m_389)
    goto __BB52;
  else
    goto __BB53;

  __BB(52):
  _224 = -s_397;
  _225 = k_388 + 18446744073709551615ul;
  _226 = nc_445(D) * _225;
  _227 = k_388 + _226;
  _228 = _227 + 18446744073709551614ul;
  _229 = _228 * 8ul;
  _230 = h_450(D) + _229;
  _231 = x_392 * _224;
  __MEM <double> (_230) = _231;
  goto __BB55;

  __BB(53):
  if (e_382 != m_389)
    goto __BB54;
  else
    goto __BB55;

  __BB(54):
  _232 = k_388 + 18446744073709551615ul;
  _233 = nc_445(D) * _232;
  _234 = k_388 + _233;
  _235 = _234 + 18446744073709551614ul;
  _236 = _235 * 8ul;
  _237 = h_450(D) + _236;
  _238 = __MEM <double> (_237);
  _239 = k_388 + 18446744073709551615ul;
  _240 = nc_445(D) * _239;
  _241 = k_388 + _240;
  _242 = _241 + 18446744073709551614ul;
  _243 = _242 * 8ul;
  _244 = h_450(D) + _243;
  _245 = -_238;
  __MEM <double> (_244) = _245;
  goto __BB55;

  __BB(55):
  p_500 = p_402 + s_397;
  x_501 = p_500 / s_397;
  y_502 = q_413 / s_397;
  z_503 = r_420 / s_397;
  q_504 = q_413 / p_500;
  r_505 = r_420 / p_500;
  j_506 = k_388;
  goto __BB59;

  __BB(56):
  _246 = k_388 + 18446744073709551615ul;
  _247 = nc_445(D) * _246;
  _248 = j_387 + _247;
  _249 = _248 + 18446744073709551615ul;
  _250 = _249 * 8ul;
  _251 = h_450(D) + _250;
  _252 = __MEM <double> (_251);
  _253 = k_388 * nc_445(D);
  _254 = j_387 + _253;
  _255 = _254 + 18446744073709551615ul;
  _256 = _255 * 8ul;
  _257 = h_450(D) + _256;
  _258 = __MEM <double> (_257);
  _259 = q_504 * _258;
  p_515 = _252 + _259;
  if (notlast_485 != 0)
    goto __BB57;
  else
    goto __BB58;

  __BB(57):
  _260 = k_388 + 1ul;
  _261 = nc_445(D) * _260;
  _262 = j_387 + _261;
  _263 = _262 + 18446744073709551615ul;
  _264 = _263 * 8ul;
  _265 = h_450(D) + _264;
  _266 = __MEM <double> (_265);
  _267 = r_505 * _266;
  p_516 = p_515 + _267;
  _268 = k_388 + 1ul;
  _269 = nc_445(D) * _268;
  _270 = j_387 + _269;
  _271 = _270 + 18446744073709551615ul;
  _272 = _271 * 8ul;
  _273 = h_450(D) + _272;
  _274 = __MEM <double> (_273);
  _275 = p_516 * z_503;
  _276 = k_388 + 1ul;
  _277 = nc_445(D) * _276;
  _278 = j_387 + _277;
  _279 = _278 + 18446744073709551615ul;
  _280 = _279 * 8ul;
  _281 = h_450(D) + _280;
  _282 = _274 - _275;
  __MEM <double> (_281) = _282;
  goto __BB58;

  __BB(58):
  p_403 = __PHI (__BB56: p_515, __BB57: p_516);
  _283 = k_388 * nc_445(D);
  _284 = j_387 + _283;
  _285 = _284 + 18446744073709551615ul;
  _286 = _285 * 8ul;
  _287 = h_450(D) + _286;
  _288 = __MEM <double> (_287);
  _289 = p_403 * y_502;
  _290 = k_388 * nc_445(D);
  _291 = j_387 + _290;
  _292 = _291 + 18446744073709551615ul;
  _293 = _292 * 8ul;
  _294 = h_450(D) + _293;
  _295 = _288 - _289;
  __MEM <double> (_294) = _295;
  _296 = k_388 + 18446744073709551615ul;
  _297 = nc_445(D) * _296;
  _298 = j_387 + _297;
  _299 = _298 + 18446744073709551615ul;
  _300 = _299 * 8ul;
  _301 = h_450(D) + _300;
  _302 = __MEM <double> (_301);
  _303 = p_403 * x_501;
  _304 = k_388 + 18446744073709551615ul;
  _305 = nc_445(D) * _304;
  _306 = j_387 + _305;
  _307 = _306 + 18446744073709551615ul;
  _308 = _307 * 8ul;
  _309 = h_450(D) + _308;
  _310 = _302 - _303;
  __MEM <double> (_309) = _310;
  j_520 = j_387 + 1ul;
  goto __BB59;

  __BB(59,loop_header(9)):
  j_387 = __PHI (__BB55: j_506, __BB58: j_520);
  p_404 = __PHI (__BB55: p_500, __BB58: p_403);
  if (j_387 <= n_423)
    goto __BB56;
  else
    goto __BB60;

  __BB(60):
  _311 = k_388 + 3ul;
  j_507 = __MIN (n_423, _311);
  i_508 = e_382;
  goto __BB64;

  __BB(61):
  _312 = i_386 + 18446744073709551615ul;
  _313 = nc_445(D) * _312;
  _314 = k_388 + _313;
  _315 = _314 + 18446744073709551615ul;
  _316 = _315 * 8ul;
  _317 = h_450(D) + _316;
  _318 = __MEM <double> (_317);
  _319 = x_501 * _318;
  _320 = i_386 + 18446744073709551615ul;
  _321 = nc_445(D) * _320;
  _322 = k_388 + _321;
  _323 = _322 * 8ul;
  _324 = h_450(D) + _323;
  _325 = __MEM <double> (_324);
  _326 = y_502 * _325;
  p_509 = _319 + _326;
  if (notlast_485 != 0)
    goto __BB62;
  else
    goto __BB63;

  __BB(62):
  _327 = i_386 + 18446744073709551615ul;
  _328 = nc_445(D) * _327;
  _329 = k_388 + _328;
  _330 = _329 + 1ul;
  _331 = _330 * 8ul;
  _332 = h_450(D) + _331;
  _333 = __MEM <double> (_332);
  _334 = z_503 * _333;
  p_510 = p_509 + _334;
  _335 = i_386 + 18446744073709551615ul;
  _336 = nc_445(D) * _335;
  _337 = k_388 + _336;
  _338 = _337 + 1ul;
  _339 = _338 * 8ul;
  _340 = h_450(D) + _339;
  _341 = __MEM <double> (_340);
  _342 = p_510 * r_505;
  _343 = i_386 + 18446744073709551615ul;
  _344 = nc_445(D) * _343;
  _345 = k_388 + _344;
  _346 = _345 + 1ul;
  _347 = _346 * 8ul;
  _348 = h_450(D) + _347;
  _349 = _341 - _342;
  __MEM <double> (_348) = _349;
  goto __BB63;

  __BB(63):
  p_405 = __PHI (__BB61: p_509, __BB62: p_510);
  _350 = i_386 + 18446744073709551615ul;
  _351 = nc_445(D) * _350;
  _352 = k_388 + _351;
  _353 = _352 * 8ul;
  _354 = h_450(D) + _353;
  _355 = __MEM <double> (_354);
  _356 = p_405 * q_504;
  _357 = i_386 + 18446744073709551615ul;
  _358 = nc_445(D) * _357;
  _359 = k_388 + _358;
  _360 = _359 * 8ul;
  _361 = h_450(D) + _360;
  _362 = _355 - _356;
  __MEM <double> (_361) = _362;
  _363 = i_386 + 18446744073709551615ul;
  _364 = nc_445(D) * _363;
  _365 = k_388 + _364;
  _366 = _365 + 18446744073709551615ul;
  _367 = _366 * 8ul;
  _368 = h_450(D) + _367;
  _369 = __MEM <double> (_368);
  _370 = i_386 + 18446744073709551615ul;
  _371 = nc_445(D) * _370;
  _372 = k_388 + _371;
  _373 = _372 + 18446744073709551615ul;
  _374 = _373 * 8ul;
  _375 = h_450(D) + _374;
  _376 = _369 - p_405;
  __MEM <double> (_375) = _376;
  i_514 = i_386 + 1ul;
  goto __BB64;

  __BB(64,loop_header(10)):
  i_386 = __PHI (__BB60: i_508, __BB63: i_514);
  p_406 = __PHI (__BB60: p_404, __BB63: p_405);
  if (i_386 <= j_507)
    goto __BB61;
  else
    goto __BB65;

  __BB(65):
  x_393 = __PHI (__BB47: x_491, __BB64: x_501);
  p_407 = __PHI (__BB47: p_486, __BB64: p_406);
  q_414 = __PHI (__BB47: q_487, __BB64: q_504);
  r_421 = __PHI (__BB47: r_490, __BB64: r_505);
  k_521 = k_388 + 1ul;
  goto __BB66;

  __BB(66,loop_header(8)):
  k_388 = __PHI (__BB41: k_484, __BB65: k_521);
  x_394 = __PHI (__BB41: x_391, __BB65: x_393);
  p_408 = __PHI (__BB41: p_401, __BB65: p_407);
  q_415 = __PHI (__BB41: q_412, __BB65: q_414);
  r_422 = __PHI (__BB41: r_419, __BB65: r_421);
  _377 = n_423 + 18446744073709551615ul;
  if (k_388 <= _377)
    goto __BB42;
  else
    goto __BB67;

  __BB(67):
  // predicted unlikely by goto predictor.
  goto __BB6;

  __BB(68):
  _424 = __PHI (__BB4: _547, __BB21: _526);
L68:
  return _424;

}


int __GIMPLE (ssa)
gsl_poly_complex_solve (const double * a, size_t n, struct gsl_poly_complex_workspace * w, double * z)
{
  double * m;
  int status;
  int D_4555;
  long unsigned int _1;
  sizetype _2;
  const double * _3;
  double _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  long unsigned int _9;
  int _10;
  int _22;
  int _24;
  int _26;
  int _28;
  int _30;
  int _32;

  __BB(2):
  if (n_12(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("number of terms must be a positive integer", ".//zsolve.c", 50, 4);
  _32 = 4;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(4):
  if (n_12(D) == 1ul)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("cannot solve for only one term", ".//zsolve.c", 55, 4);
  _30 = 4;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(6):
  _1 = n_12(D) * 8ul;
  _2 = _1 + 18446744073709551608ul;
  _3 = a_13(D) + _2;
  _4 = __MEM <const double> (_3);
  if (_4 == 0.0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_error ("leading term of polynomial must be non-zero", ".//zsolve.c", 60, 4);
  _28 = 4;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(8):
  _5 = w_15(D)->nc;
  _6 = n_12(D) + 18446744073709551615ul;
  if (_5 != _6)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  gsl_error ("size of workspace does not match polynomial", ".//zsolve.c", 65, 4);
  _26 = 4;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(10):
  m_16 = w_15(D)->matrix;
  _7 = n_12(D) + 18446744073709551615ul;
  set_companion_matrix (a_13(D), _7, m_16);
  _8 = n_12(D) + 18446744073709551615ul;
  balance_companion_matrix (m_16, _8);
  _9 = n_12(D) + 18446744073709551615ul;
  status_21 = qr_companion (m_16, _9, z_19(D));
  if (status_21 != 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  gsl_error ("root solving qr method failed to converge", ".//zsolve.c", 78, 5);
  _24 = 5;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(12):
  _22 = 0;
  goto __BB13;

  __BB(13):
  _10 = __PHI (__BB3: _32, __BB5: _30, __BB7: _28, __BB9: _26, __BB11: _24, __BB12: _22);
L10:
  return _10;

}


