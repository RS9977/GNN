int __GIMPLE (ssa)
gsl_sum_levin_u_step (const double term, const size_t n, const size_t nmax, struct gsl_sum_levin_u_workspace * w, double * sum_accel)
{
  double c;
  int j;
  unsigned int i;
  double ratio;
  double factor;
  double result;
  int D_4386;
  double * _1;
  double _2;
  double * _3;
  double _4;
  double * _5;
  double _6;
  double * _7;
  double * _8;
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
  double * _20;
  long unsigned int _21;
  double * _22;
  double _23;
  double _24;
  double * _25;
  long unsigned int _26;
  double * _27;
  double _28;
  double * _29;
  long unsigned int _30;
  double * _31;
  double _32;
  double * _33;
  long unsigned int _34;
  long unsigned int _35;
  long unsigned int _36;
  long unsigned int _37;
  long unsigned int _38;
  double * _39;
  double * _40;
  long unsigned int _41;
  double * _42;
  double * _43;
  long unsigned int _44;
  long unsigned int _45;
  long unsigned int _46;
  long unsigned int _47;
  long unsigned int _48;
  double * _49;
  double _50;
  long unsigned int _51;
  double * _52;
  long unsigned int _53;
  double * _54;
  double _55;
  double _56;
  double * _57;
  long unsigned int _58;
  long unsigned int _59;
  long unsigned int _60;
  double * _61;
  double _62;
  double * _63;
  long unsigned int _64;
  double * _65;
  double _66;
  double _67;
  double * _68;
  long unsigned int _69;
  long unsigned int _70;
  long unsigned int _71;
  double * _72;
  double _73;
  double _74;
  double * _75;
  long unsigned int _76;
  long unsigned int _77;
  long unsigned int _78;
  double * _79;
  double _80;
  unsigned int _81;
  unsigned int _82;
  int _83;
  double _84;
  double _85;
  long unsigned int _86;
  double _87;
  double * _88;
  sizetype _89;
  sizetype _90;
  sizetype _91;
  double * _92;
  double _93;
  double * _94;
  long unsigned int _95;
  long unsigned int _96;
  double * _97;
  double _98;
  double _99;
  double * _100;
  long unsigned int _101;
  long unsigned int _102;
  double * _103;
  double _104;
  double * _105;
  sizetype _106;
  sizetype _107;
  sizetype _108;
  double * _109;
  double _110;
  double * _111;
  long unsigned int _112;
  long unsigned int _113;
  double * _114;
  double _115;
  double _116;
  double * _117;
  long unsigned int _118;
  long unsigned int _119;
  double * _120;
  double _121;
  double * _122;
  long unsigned int _123;
  long unsigned int _124;
  long unsigned int _125;
  int _126;
  long unsigned int _127;
  long unsigned int _128;
  long unsigned int _129;
  double * _130;
  double _131;
  double * _132;
  long unsigned int _133;
  long unsigned int _134;
  long unsigned int _135;
  long unsigned int _136;
  long unsigned int _137;
  long unsigned int _138;
  double * _139;
  double _140;
  double _141;
  double * _142;
  long unsigned int _143;
  long unsigned int _144;
  long unsigned int _145;
  long unsigned int _146;
  long unsigned int _147;
  long unsigned int _148;
  double * _149;
  double _150;
  double * _151;
  long unsigned int _152;
  long unsigned int _153;
  long unsigned int _154;
  int _155;
  long unsigned int _156;
  long unsigned int _157;
  long unsigned int _158;
  double * _159;
  double _160;
  double * _161;
  long unsigned int _162;
  long unsigned int _163;
  long unsigned int _164;
  long unsigned int _165;
  long unsigned int _166;
  long unsigned int _167;
  double * _168;
  double _169;
  double _170;
  double * _171;
  long unsigned int _172;
  long unsigned int _173;
  long unsigned int _174;
  long unsigned int _175;
  long unsigned int _176;
  long unsigned int _177;
  double * _178;
  double _179;
  long unsigned int _180;
  double * _181;
  long unsigned int _182;
  long unsigned int _183;
  int _184;
  long unsigned int _185;
  long unsigned int _186;
  long unsigned int _187;
  double * _188;
  double * _189;
  long unsigned int _190;
  long unsigned int _191;
  long unsigned int _192;
  long unsigned int _193;
  long unsigned int _194;
  double * _195;
  double _196;
  double * _197;
  long unsigned int _198;
  long unsigned int _199;
  int _200;
  long unsigned int _201;
  long unsigned int _202;
  long unsigned int _203;
  double * _204;
  double * _205;
  long unsigned int _206;
  long unsigned int _207;
  long unsigned int _208;
  long unsigned int _209;
  long unsigned int _210;
  double * _211;
  double _212;
  double * _213;
  double _214;
  double * _215;
  double _216;
  double * _217;
  long unsigned int _218;
  long unsigned int _219;
  long unsigned int _220;
  long unsigned int _221;
  double * _222;
  double _223;
  double * _224;
  long unsigned int _225;
  long unsigned int _226;
  long unsigned int _227;
  long unsigned int _228;
  double * _229;
  double _230;
  double _231;
  double _232;
  double * _233;
  double _234;
  double * _235;
  long unsigned int _236;
  long unsigned int _237;
  double * _238;
  double _239;
  long unsigned int _240;
  int _246;
  int _270;
  int _294;

  __BB(2):
  if (n_252(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  __MEM <double> (sum_accel_267(D)) = term_257(D);
  w_256(D)->sum_plain = term_257(D);
  _1 = w_256(D)->q_den;
  _2 = 1.0e+0 / term_257(D);
  __MEM <double> (_1) = _2;
  _3 = w_256(D)->q_num;
  __MEM <double> (_3) = 1.0e+0;
  _4 = term_257(D) * term_257(D);
  _5 = w_256(D)->dq_den;
  _6 = -1.0e+0 / _4;
  __MEM <double> (_5) = _6;
  _7 = w_256(D)->dq_num;
  __MEM <double> (_7) = 0.0;
  _8 = w_256(D)->dsum;
  __MEM <double> (_8) = 1.0e+0;
  _294 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(4):
  factor_253 = 1.0e+0;
  _9 = (double) n_252(D);
  _10 = (double) n_252(D);
  _11 = _10 + 1.0e+0;
  ratio_254 = _9 / _11;
  _12 = w_256(D)->sum_plain;
  _13 = term_257(D) + _12;
  w_256(D)->sum_plain = _13;
  _14 = (double) n_252(D);
  _15 = _14 + 1.0e+0;
  _16 = term_257(D) * _15;
  _17 = (double) n_252(D);
  _18 = _17 + 1.0e+0;
  _19 = _16 * _18;
  _20 = w_256(D)->q_den;
  _21 = n_252(D) * 8ul;
  _22 = _20 + _21;
  _23 = 1.0e+0 / _19;
  __MEM <double> (_22) = _23;
  _24 = w_256(D)->sum_plain;
  _25 = w_256(D)->q_den;
  _26 = n_252(D) * 8ul;
  _27 = _25 + _26;
  _28 = __MEM <double> (_27);
  _29 = w_256(D)->q_num;
  _30 = n_252(D) * 8ul;
  _31 = _29 + _30;
  _32 = _24 * _28;
  __MEM <double> (_31) = _32;
  i_261 = 0u;
  goto __BB6;

  __BB(5):
  _33 = w_256(D)->dq_den;
  _34 = (long unsigned int) i_242;
  _35 = nmax_262(D) + 1ul;
  _36 = _34 * _35;
  _37 = n_252(D) + _36;
  _38 = _37 * 8ul;
  _39 = _33 + _38;
  __MEM <double> (_39) = 0.0;
  _40 = w_256(D)->q_den;
  _41 = n_252(D) * 8ul;
  _42 = _40 + _41;
  _43 = w_256(D)->dq_num;
  _44 = (long unsigned int) i_242;
  _45 = nmax_262(D) + 1ul;
  _46 = _44 * _45;
  _47 = n_252(D) + _46;
  _48 = _47 * 8ul;
  _49 = _43 + _48;
  _50 = __MEM <double> (_42);
  __MEM <double> (_49) = _50;
  i_286 = i_242 + 1u;
  goto __BB6;

  __BB(6,loop_header(1)):
  i_242 = __PHI (__BB4: i_261, __BB5: i_286);
  _51 = (long unsigned int) i_242;
  if (n_252(D) > _51)
    goto __BB5;
  else
    goto __BB7;

  __BB(7):
  _52 = w_256(D)->q_den;
  _53 = n_252(D) * 8ul;
  _54 = _52 + _53;
  _55 = __MEM <double> (_54);
  _56 = -_55;
  _57 = w_256(D)->dq_den;
  _58 = nmax_262(D) + 2ul;
  _59 = n_252(D) * _58;
  _60 = _59 * 8ul;
  _61 = _57 + _60;
  _62 = _56 / term_257(D);
  __MEM <double> (_61) = _62;
  _63 = w_256(D)->q_den;
  _64 = n_252(D) * 8ul;
  _65 = _63 + _64;
  _66 = __MEM <double> (_65);
  _67 = w_256(D)->sum_plain;
  _68 = w_256(D)->dq_den;
  _69 = nmax_262(D) + 2ul;
  _70 = n_252(D) * _69;
  _71 = _70 * 8ul;
  _72 = _68 + _71;
  _73 = __MEM <double> (_72);
  _74 = _67 * _73;
  _75 = w_256(D)->dq_num;
  _76 = nmax_262(D) + 2ul;
  _77 = n_252(D) * _76;
  _78 = _77 * 8ul;
  _79 = _75 + _78;
  _80 = _66 + _74;
  __MEM <double> (_79) = _80;
  _81 = (unsigned int) n_252(D);
  _82 = _81 + 4294967295u;
  j_265 = (int) _82;
  goto __BB12;

  __BB(8):
  _83 = j_245 + 1;
  _84 = (double) _83;
  _85 = factor_241 * _84;
  _86 = n_252(D) + 1ul;
  _87 = (double) _86;
  c_273 = _85 / _87;
  factor_274 = factor_241 * ratio_254;
  _88 = w_256(D)->q_den;
  _89 = (sizetype) j_245;
  _90 = _89 + 1ul;
  _91 = _90 * 8ul;
  _92 = _88 + _91;
  _93 = __MEM <double> (_92);
  _94 = w_256(D)->q_den;
  _95 = (long unsigned int) j_245;
  _96 = _95 * 8ul;
  _97 = _94 + _96;
  _98 = __MEM <double> (_97);
  _99 = c_273 * _98;
  _100 = w_256(D)->q_den;
  _101 = (long unsigned int) j_245;
  _102 = _101 * 8ul;
  _103 = _100 + _102;
  _104 = _93 - _99;
  __MEM <double> (_103) = _104;
  _105 = w_256(D)->q_num;
  _106 = (sizetype) j_245;
  _107 = _106 + 1ul;
  _108 = _107 * 8ul;
  _109 = _105 + _108;
  _110 = __MEM <double> (_109);
  _111 = w_256(D)->q_num;
  _112 = (long unsigned int) j_245;
  _113 = _112 * 8ul;
  _114 = _111 + _113;
  _115 = __MEM <double> (_114);
  _116 = c_273 * _115;
  _117 = w_256(D)->q_num;
  _118 = (long unsigned int) j_245;
  _119 = _118 * 8ul;
  _120 = _117 + _119;
  _121 = _110 - _116;
  __MEM <double> (_120) = _121;
  i_277 = 0u;
  goto __BB10;

  __BB(9):
  _122 = w_256(D)->dq_den;
  _123 = (long unsigned int) i_243;
  _124 = nmax_262(D) + 1ul;
  _125 = _123 * _124;
  _126 = j_245 + 1;
  _127 = (long unsigned int) _126;
  _128 = _125 + _127;
  _129 = _128 * 8ul;
  _130 = _122 + _129;
  _131 = __MEM <double> (_130);
  _132 = w_256(D)->dq_den;
  _133 = (long unsigned int) i_243;
  _134 = nmax_262(D) + 1ul;
  _135 = _133 * _134;
  _136 = (long unsigned int) j_245;
  _137 = _135 + _136;
  _138 = _137 * 8ul;
  _139 = _132 + _138;
  _140 = __MEM <double> (_139);
  _141 = c_273 * _140;
  _142 = w_256(D)->dq_den;
  _143 = (long unsigned int) i_243;
  _144 = nmax_262(D) + 1ul;
  _145 = _143 * _144;
  _146 = (long unsigned int) j_245;
  _147 = _145 + _146;
  _148 = _147 * 8ul;
  _149 = _142 + _148;
  _150 = _131 - _141;
  __MEM <double> (_149) = _150;
  _151 = w_256(D)->dq_num;
  _152 = (long unsigned int) i_243;
  _153 = nmax_262(D) + 1ul;
  _154 = _152 * _153;
  _155 = j_245 + 1;
  _156 = (long unsigned int) _155;
  _157 = _154 + _156;
  _158 = _157 * 8ul;
  _159 = _151 + _158;
  _160 = __MEM <double> (_159);
  _161 = w_256(D)->dq_num;
  _162 = (long unsigned int) i_243;
  _163 = nmax_262(D) + 1ul;
  _164 = _162 * _163;
  _165 = (long unsigned int) j_245;
  _166 = _164 + _165;
  _167 = _166 * 8ul;
  _168 = _161 + _167;
  _169 = __MEM <double> (_168);
  _170 = c_273 * _169;
  _171 = w_256(D)->dq_num;
  _172 = (long unsigned int) i_243;
  _173 = nmax_262(D) + 1ul;
  _174 = _172 * _173;
  _175 = (long unsigned int) j_245;
  _176 = _174 + _175;
  _177 = _176 * 8ul;
  _178 = _171 + _177;
  _179 = _160 - _170;
  __MEM <double> (_178) = _179;
  i_283 = i_243 + 1u;
  goto __BB10;

  __BB(10,loop_header(4)):
  i_243 = __PHI (__BB8: i_277, __BB9: i_283);
  _180 = (long unsigned int) i_243;
  if (n_252(D) > _180)
    goto __BB9;
  else
    goto __BB11;

  __BB(11):
  _181 = w_256(D)->dq_den;
  _182 = nmax_262(D) + 1ul;
  _183 = n_252(D) * _182;
  _184 = j_245 + 1;
  _185 = (long unsigned int) _184;
  _186 = _183 + _185;
  _187 = _186 * 8ul;
  _188 = _181 + _187;
  _189 = w_256(D)->dq_den;
  _190 = nmax_262(D) + 1ul;
  _191 = n_252(D) * _190;
  _192 = (long unsigned int) j_245;
  _193 = _191 + _192;
  _194 = _193 * 8ul;
  _195 = _189 + _194;
  _196 = __MEM <double> (_188);
  __MEM <double> (_195) = _196;
  _197 = w_256(D)->dq_num;
  _198 = nmax_262(D) + 1ul;
  _199 = n_252(D) * _198;
  _200 = j_245 + 1;
  _201 = (long unsigned int) _200;
  _202 = _199 + _201;
  _203 = _202 * 8ul;
  _204 = _197 + _203;
  _205 = w_256(D)->dq_num;
  _206 = nmax_262(D) + 1ul;
  _207 = n_252(D) * _206;
  _208 = (long unsigned int) j_245;
  _209 = _207 + _208;
  _210 = _209 * 8ul;
  _211 = _205 + _210;
  _212 = __MEM <double> (_204);
  __MEM <double> (_211) = _212;
  j_280 = j_245 + _Literal (int) -1;
  goto __BB12;

  __BB(12,loop_header(2)):
  factor_241 = __PHI (__BB7: factor_253, __BB11: factor_274);
  j_245 = __PHI (__BB7: j_265, __BB11: j_280);
  if (j_245 >= 0)
    goto __BB8;
  else
    goto __BB13;

  __BB(13):
  _213 = w_256(D)->q_num;
  _214 = __MEM <double> (_213);
  _215 = w_256(D)->q_den;
  _216 = __MEM <double> (_215);
  result_266 = _214 / _216;
  __MEM <double> (sum_accel_267(D)) = result_266;
  i_269 = 0u;
  goto __BB15;

  __BB(14):
  _217 = w_256(D)->dq_num;
  _218 = (long unsigned int) i_244;
  _219 = nmax_262(D) + 1ul;
  _220 = _218 * _219;
  _221 = _220 * 8ul;
  _222 = _217 + _221;
  _223 = __MEM <double> (_222);
  _224 = w_256(D)->dq_den;
  _225 = (long unsigned int) i_244;
  _226 = nmax_262(D) + 1ul;
  _227 = _225 * _226;
  _228 = _227 * 8ul;
  _229 = _224 + _228;
  _230 = __MEM <double> (_229);
  _231 = result_266 * _230;
  _232 = _223 - _231;
  _233 = w_256(D)->q_den;
  _234 = __MEM <double> (_233);
  _235 = w_256(D)->dsum;
  _236 = (long unsigned int) i_244;
  _237 = _236 * 8ul;
  _238 = _235 + _237;
  _239 = _232 / _234;
  __MEM <double> (_238) = _239;
  i_272 = i_244 + 1u;
  goto __BB15;

  __BB(15,loop_header(3)):
  i_244 = __PHI (__BB13: i_269, __BB14: i_272);
  _240 = (long unsigned int) i_244;
  if (n_252(D) >= _240)
    goto __BB14;
  else
    goto __BB16;

  __BB(16):
  _270 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB17;

  __BB(17):
  _246 = __PHI (__BB3: _294, __BB16: _270);
L14:
  return _246;

}


int __GIMPLE (ssa)
gsl_sum_levin_u_minmax (const double * array, const size_t array_size, const size_t min_terms, const size_t max_terms, struct gsl_sum_levin_u_workspace * w, double * sum_accel, double * abserr)
{
  double dn;
  const double t;
  double dn;
  const double t;
  double least_trunc_result;
  double least_trunc_noise;
  double least_trunc;
  int converging;
  int before;
  int better;
  unsigned int i;
  size_t n;
  double variance;
  double result_nm1;
  double result_n;
  double actual_trunc_nm1;
  double actual_trunc_n;
  double trunc_nm1;
  double trunc_n;
  double noise_nm1;
  double noise_n;
  const size_t nmax;
  const double SMALL;
  size_t size;
  int iftmp.3;
  int iftmp.1;
  int D_4358;
  long unsigned int _1;
  sizetype _2;
  const double * _3;
  double _4;
  double _5;
  double _6;
  long unsigned int _7;
  long unsigned int _8;
  const double * _9;
  double * _10;
  long unsigned int _11;
  long unsigned int _12;
  double * _13;
  double _14;
  double _15;
  long unsigned int _16;
  long unsigned int _17;
  const double * _18;
  double _19;
  double _20;
  long unsigned int _21;
  long unsigned int _22;
  const double * _23;
  double _24;
  double _25;
  double _26;
  double * _27;
  long unsigned int _28;
  long unsigned int _29;
  double * _30;
  double _31;
  double _32;
  long unsigned int _33;
  long unsigned int _34;
  const double * _35;
  double _36;
  double _37;
  long unsigned int _38;
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
  int _76;
  int _77;
  int _78;
  int _79;
  int _127;
  int _128;
  int _130;
  int _131;
  int _147;
  int _152;
  int _165;
  int _170;

  __BB(2):
  size_86 = array_size_85(D);
  goto __BB4;

  __BB(3):
  size_89 = size_49 + 18446744073709551615ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  size_49 = __PHI (__BB2: size_86, __BB3: size_89);
  if (size_49 != 0ul)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _1 = size_49 * 8ul;
  _2 = _1 + 18446744073709551608ul;
  _3 = array_87(D) + _2;
  _4 = __MEM <const double> (_3);
  if (_4 == 0.0)
    goto __BB3;
  else
    goto __BB6;

  __BB(6):
  if (size_49 == 0ul)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  __MEM <double> (sum_accel_141(D)) = 0.0;
  __MEM <double> (abserr_144(D)) = 0.0;
  w_116(D)->sum_plain = 0.0;
  w_116(D)->terms_used = 0ul;
  _170 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB42;

  __BB(8):
  if (size_49 == 1ul)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  _5 = __MEM <const double> (array_87(D));
  __MEM <double> (sum_accel_141(D)) = _5;
  __MEM <double> (abserr_144(D)) = 0.0;
  _6 = __MEM <const double> (array_87(D));
  w_116(D)->sum_plain = _6;
  w_116(D)->terms_used = 1ul;
  _165 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB42;

  __BB(10):
  SMALL_90 = 1.00000000000000002081668171172168513294309377670288085938e-2;
  _7 = __MAX (array_size_85(D), max_terms_91(D));
  nmax_92 = _7 + 18446744073709551615ul;
  noise_n_93 = 0.0;
  noise_nm1_94 = 0.0;
  trunc_n_95 = 0.0;
  trunc_nm1_96 = 0.0;
  actual_trunc_n_97 = 0.0;
  actual_trunc_nm1_98 = 0.0;
  result_n = 0.0;
  result_nm1_100 = 0.0;
  variance_101 = 0.0;
  better_102 = 0;
  before_103 = 0;
  converging_104 = 0;
  least_trunc_105 = 1.79769313486231570814527423731704356798070567525844996599e+308;
  least_trunc_noise_106 = 1.79769313486231570814527423731704356798070567525844996599e+308;
  n_107 = 0ul;
  goto __BB12;

  __BB(11):
  _8 = n_57 * 8ul;
  _9 = array_87(D) + _8;
  t_157 = __MEM <const double> (_9);
  result_nm1_158 = result_n;
  gsl_sum_levin_u_step (t_157, n_57, nmax_92, w_116(D), &result_n);
  n_160 = n_57 + 1ul;
  goto __BB12;

  __BB(12,loop_header(2)):
  n_57 = __PHI (__BB10: n_107, __BB11: n_160);
  if (n_57 < min_terms_108(D))
    goto __BB11;
  else
    goto __BB13;

  __BB(13):
  least_trunc_result_109 = result_n;
  variance_110 = 0.0;
  i_111 = 0u;
  goto __BB15;

  __BB(14):
  _10 = w_116(D)->dsum;
  _11 = (long unsigned int) i_59;
  _12 = _11 * 8ul;
  _13 = _10 + _12;
  _14 = __MEM <double> (_13);
  _15 = _14 * 2.220446049250313080847263336181640625e-16;
  _16 = (long unsigned int) i_59;
  _17 = _16 * 8ul;
  _18 = array_87(D) + _17;
  _19 = __MEM <const double> (_18);
  dn_154 = _15 * _19;
  _20 = dn_154 * dn_154;
  variance_155 = variance_55 + _20;
  i_156 = i_59 + 1u;
  goto __BB15;

  __BB(15,loop_header(3)):
  variance_55 = __PHI (__BB13: variance_110, __BB14: variance_155);
  i_59 = __PHI (__BB13: i_111, __BB14: i_156);
  _21 = (long unsigned int) i_59;
  if (n_57 > _21)
    goto __BB14;
  else
    goto __BB16;

  __BB(16):
  noise_n_113 = sqrt (variance_55);
  goto __BB37;

  __BB(17):
  _22 = n_58 * 8ul;
  _23 = array_87(D) + _22;
  t_114 = __MEM <const double> (_23);
  result_nm1_115 = result_n;
  gsl_sum_levin_u_step (t_114, n_58, nmax_92, w_116(D), &result_n);
  actual_trunc_nm1_118 = actual_trunc_n_54;
  _24 = result_n;
  _25 = _24 - result_nm1_115;
  actual_trunc_n_119 = __ABS _25;
  trunc_nm1_120 = trunc_n_52;
  _26 = actual_trunc_n_119 + actual_trunc_nm1_118;
  trunc_n_121 = _26 * 5.0e-1;
  noise_nm1_122 = noise_n_50;
  variance_123 = 0.0;
  i_124 = 0u;
  goto __BB19;

  __BB(18):
  _27 = w_116(D)->dsum;
  _28 = (long unsigned int) i_60;
  _29 = _28 * 8ul;
  _30 = _27 + _29;
  _31 = __MEM <double> (_30);
  _32 = _31 * 2.220446049250313080847263336181640625e-16;
  _33 = (long unsigned int) i_60;
  _34 = _33 * 8ul;
  _35 = array_87(D) + _34;
  _36 = __MEM <const double> (_35);
  dn_138 = _32 * _36;
  _37 = dn_138 * dn_138;
  variance_139 = variance_56 + _37;
  i_140 = i_60 + 1u;
  goto __BB19;

  __BB(19,loop_header(5)):
  variance_56 = __PHI (__BB17: variance_123, __BB18: variance_139);
  i_60 = __PHI (__BB17: i_124, __BB18: i_140);
  _38 = (long unsigned int) i_60;
  if (n_58 >= _38)
    goto __BB18;
  else
    goto __BB20;

  __BB(20):
  noise_n_126 = sqrt (variance_56);
  if (trunc_n_121 < trunc_nm1_120)
    goto __BB22;
  else
    goto __BB21;

  __BB(21):
  _39 = result_n;
  _40 = __ABS _39;
  _41 = SMALL_90 * _40;
  if (trunc_n_121 < _41)
    goto __BB22;
  else
    goto __BB23;

  __BB(22):
  _128 = 1;
  goto __BB24;

  __BB(23):
  _127 = 0;
  goto __BB24;

  __BB(24):
  _78 = __PHI (__BB22: _128, __BB23: _127);
  better_129 = _78;
  if (converging_62 != 0)
    goto __BB27;
  else
    goto __BB25;

  __BB(25):
  if (better_129 != 0)
    goto __BB26;
  else
    goto __BB28;

  __BB(26):
  if (before_61 != 0)
    goto __BB27;
  else
    goto __BB28;

  __BB(27):
  _131 = 1;
  goto __BB29;

  __BB(28):
  _130 = 0;
  goto __BB29;

  __BB(29):
  _79 = __PHI (__BB27: _131, __BB28: _130);
  converging_132 = _79;
  before_133 = better_129;
  if (converging_132 != 0)
    goto __BB30;
  else
    goto __BB36;

  __BB(30):
  if (trunc_n_121 < least_trunc_66)
    goto __BB31;
  else
    goto __BB32;

  __BB(31):
  least_trunc_result_134 = result_n;
  least_trunc_135 = trunc_n_121;
  least_trunc_noise_136 = noise_n_126;
  goto __BB32;

  __BB(32):
  least_trunc_64 = __PHI (__BB30: least_trunc_66, __BB31: least_trunc_135);
  least_trunc_noise_68 = __PHI (__BB30: least_trunc_noise_70, __BB31: least_trunc_noise_136);
  least_trunc_result_72 = __PHI (__BB30: least_trunc_result_74, __BB31: least_trunc_result_134);
  _42 = trunc_n_121 / 3.0e+0;
  if (noise_n_126 > _42)
    goto __BB33;
  else
    goto __BB34;

  __BB(33):
  goto __BB38;

  __BB(34):
  _43 = result_n;
  _44 = __ABS _43;
  _45 = _44 * 2.220446049250313080847263336181640625e-15;
  if (trunc_n_121 < _45)
    goto __BB35;
  else
    goto __BB36;

  __BB(35):
  goto __BB38;

  __BB(36):
  least_trunc_65 = __PHI (__BB29: least_trunc_66, __BB34: least_trunc_64);
  least_trunc_noise_69 = __PHI (__BB29: least_trunc_noise_70, __BB34: least_trunc_noise_68);
  least_trunc_result_73 = __PHI (__BB29: least_trunc_result_74, __BB34: least_trunc_result_72);
  n_137 = n_58 + 1ul;
  goto __BB37;

  __BB(37,loop_header(4)):
  noise_n_50 = __PHI (__BB16: noise_n_113, __BB36: noise_n_126);
  trunc_n_52 = __PHI (__BB16: trunc_n_95, __BB36: trunc_n_121);
  actual_trunc_n_54 = __PHI (__BB16: actual_trunc_n_97, __BB36: actual_trunc_n_119);
  n_58 = __PHI (__BB16: n_57, __BB36: n_137);
  before_61 = __PHI (__BB16: before_103, __BB36: before_133);
  converging_62 = __PHI (__BB16: converging_104, __BB36: converging_132);
  least_trunc_66 = __PHI (__BB16: least_trunc_105, __BB36: least_trunc_65);
  least_trunc_noise_70 = __PHI (__BB16: least_trunc_noise_106, __BB36: least_trunc_noise_69);
  least_trunc_result_74 = __PHI (__BB16: least_trunc_result_109, __BB36: least_trunc_result_73);
  if (n_58 <= nmax_92)
    goto __BB17;
  else
    goto __BB38;

  __BB(38):
  noise_n_51 = __PHI (__BB33: noise_n_126, __BB35: noise_n_126, __BB37: noise_n_50);
  trunc_n_53 = __PHI (__BB33: trunc_n_121, __BB35: trunc_n_121, __BB37: trunc_n_52);
  converging_63 = __PHI (__BB33: converging_132, __BB35: converging_132, __BB37: converging_62);
  least_trunc_67 = __PHI (__BB33: least_trunc_64, __BB35: least_trunc_64, __BB37: least_trunc_66);
  least_trunc_noise_71 = __PHI (__BB33: least_trunc_noise_68, __BB35: least_trunc_noise_68, __BB37: least_trunc_noise_70);
  least_trunc_result_75 = __PHI (__BB33: least_trunc_result_72, __BB35: least_trunc_result_72, __BB37: least_trunc_result_74);
  if (converging_63 != 0)
    goto __BB39;
  else
    goto __BB40;

  __BB(39):
  __MEM <double> (sum_accel_141(D)) = least_trunc_result_75;
  _46 = GSL_MAX_DBL (least_trunc_67, least_trunc_noise_71);
  __MEM <double> (abserr_144(D)) = _46;
  w_116(D)->terms_used = n_58;
  _152 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB41;

  __BB(40):
  _47 = result_n;
  __MEM <double> (sum_accel_141(D)) = _47;
  _48 = GSL_MAX_DBL (trunc_n_53, noise_n_51);
  __MEM <double> (abserr_144(D)) = _48;
  w_116(D)->terms_used = n_58;
  _147 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB41;

  __BB(41):
  _76 = __PHI (__BB39: _152, __BB40: _147);
  result_n ={v} _Literal (double) {CLOBBER};
  goto __BB42;

  __BB(42):
  _77 = __PHI (__BB7: _170, __BB9: _165, __BB41: _76);
L40:
  return _77;

}


int __GIMPLE (ssa)
gsl_sum_levin_u_accel (const double * array, const size_t array_size, struct gsl_sum_levin_u_workspace * w, double * sum_accel, double * abserr)
{
  int D_4353;
  long unsigned int _1;
  int _9;

  __BB(2):
  _1 = array_size_2(D) + 18446744073709551615ul;
  _9 = gsl_sum_levin_u_minmax (array_4(D), array_size_2(D), 0ul, _1, w_5(D), sum_accel_6(D), abserr_7(D));
  goto __BB3;

  __BB(3):
L0:
  return _9;

}


