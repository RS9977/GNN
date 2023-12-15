void __GIMPLE (ssa)
increment_state (struct ranlxs_state_t * state)
{
  unsigned int jr;
  unsigned int ir;
  double carry;
  double * ydbl;
  double * xdbl;
  float * xflt;
  double y3;
  double y2;
  double y1;
  double x;
  int m;
  int kmax;
  int k;
  long unsigned int _1;
  long unsigned int _2;
  double * _3;
  double _4;
  long unsigned int _5;
  long unsigned int _6;
  double * _7;
  double _8;
  long unsigned int _9;
  long unsigned int _10;
  double * _11;
  int _12;
  int _13;
  unsigned int _14;
  unsigned int _15;
  double * _16;
  double _17;
  double _18;
  double * _19;
  double _20;
  double * _21;
  double _22;
  double _23;
  double * _24;
  double _25;
  double * _26;
  double _27;
  double _28;
  double * _29;
  double * _30;
  double _31;
  double * _32;
  double _33;
  double _34;
  double * _35;
  double * _36;
  double _37;
  double * _38;
  double _39;
  double _40;
  double * _41;
  double _42;
  double * _43;
  double _44;
  double _45;
  double * _46;
  double * _47;
  double _48;
  double * _49;
  double _50;
  double _51;
  double * _52;
  double * _53;
  double _54;
  double * _55;
  double _56;
  double _57;
  double * _58;
  double * _59;
  double _60;
  double * _61;
  double _62;
  double _63;
  double * _64;
  double * _65;
  double _66;
  double * _67;
  double _68;
  double _69;
  double * _70;
  double * _71;
  double _72;
  double * _73;
  double _74;
  double _75;
  double * _76;
  double * _77;
  double _78;
  double * _79;
  double _80;
  double _81;
  double * _82;
  double * _83;
  unsigned int _84;
  long unsigned int _85;
  long unsigned int _86;
  double * _87;
  double _88;
  long unsigned int _89;
  long unsigned int _90;
  double * _91;
  double _92;
  long unsigned int _93;
  long unsigned int _94;
  double * _95;
  double _96;
  long unsigned int _97;
  long unsigned int _98;
  double * _99;
  double _100;
  int _101;
  int _102;
  long unsigned int _103;
  long unsigned int _104;
  double * _105;
  double _106;
  double _107;
  long unsigned int _108;
  long unsigned int _109;
  double * _110;
  double _111;
  long unsigned int _112;
  long unsigned int _113;
  double * _114;
  double _115;
  double _116;
  long unsigned int _117;
  long unsigned int _118;
  double * _119;
  double _120;
  long unsigned int _121;
  long unsigned int _122;
  double * _123;
  long unsigned int _124;
  long unsigned int _125;
  double * _126;
  double _127;
  double _128;
  double _129;
  double _130;
  double _131;
  int _132;
  long unsigned int _133;
  long unsigned int _134;
  float * _135;
  float _136;
  int _137;
  long unsigned int _138;
  long unsigned int _139;
  float * _140;
  float _141;
  unsigned int _142;
  unsigned int _143;

  __BB(2):
  xflt_192 = &state_191(D)->xflt;
  xdbl_193 = &state_191(D)->xdbl;
  ydbl_194 = &state_191(D)->ydbl;
  carry_196 = state_191(D)->carry;
  ir_197 = state_191(D)->ir;
  jr_198 = state_191(D)->jr;
  k_199 = 0;
  goto __BB7;

  __BB(3):
  _1 = (long unsigned int) jr_184;
  _2 = _1 * 8ul;
  _3 = xdbl_193 + _2;
  _4 = __MEM <double> (_3);
  _5 = (long unsigned int) ir_182;
  _6 = _5 * 8ul;
  _7 = xdbl_193 + _6;
  _8 = __MEM <double> (_7);
  y1_283 = _4 - _8;
  y2_284 = y1_283 - carry_177;
  if (y2_284 < 0.0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  carry_286 = 3.552713678800500929355621337890625e-15;
  y2_287 = y2_284 + 1.0e+0;
  goto __BB6;

  __BB(5):
  carry_285 = 0.0;
  goto __BB6;

  __BB(6):
  y2_157 = __PHI (__BB4: y2_287, __BB5: y2_284);
  carry_176 = __PHI (__BB4: carry_286, __BB5: carry_285);
  _9 = (long unsigned int) ir_182;
  _10 = _9 * 8ul;
  _11 = xdbl_193 + _10;
  __MEM <double> (_11) = y2_157;
  _12 = next[ir_182];
  ir_289 = (unsigned int) _12;
  _13 = next[jr_184];
  jr_290 = (unsigned int) _13;
  k_291 = k_144 + 1;
  goto __BB7;

  __BB(7,loop_header(1)):
  k_144 = __PHI (__BB2: k_199, __BB6: k_291);
  carry_177 = __PHI (__BB2: carry_196, __BB6: carry_176);
  ir_182 = __PHI (__BB2: ir_197, __BB6: ir_289);
  jr_184 = __PHI (__BB2: jr_198, __BB6: jr_290);
  if (ir_182 != 0u)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  _14 = state_191(D)->pr;
  _15 = _14 + 4294967284u;
  kmax_200 = (int) _15;
  goto __BB35;

  __BB(9):
  _16 = xdbl_193 + 56ul;
  _17 = __MEM <double> (_16);
  _18 = __MEM <double> (xdbl_193);
  y1_232 = _17 - _18;
  y1_233 = y1_232 - carry_179;
  _19 = xdbl_193 + 64ul;
  _20 = __MEM <double> (_19);
  _21 = xdbl_193 + 8ul;
  _22 = __MEM <double> (_21);
  y2_234 = _20 - _22;
  if (y1_233 < 0.0)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  _23 = 3.552713678800500929355621337890625e-15;
  y2_235 = y2_234 - _23;
  y1_236 = y1_233 + 1.0e+0;
  goto __BB11;

  __BB(11):
  y1_150 = __PHI (__BB9: y1_233, __BB10: y1_236);
  y2_158 = __PHI (__BB9: y2_234, __BB10: y2_235);
  __MEM <double> (xdbl_193) = y1_150;
  _24 = xdbl_193 + 72ul;
  _25 = __MEM <double> (_24);
  _26 = xdbl_193 + 16ul;
  _27 = __MEM <double> (_26);
  y3_238 = _25 - _27;
  if (y2_158 < 0.0)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  _28 = 3.552713678800500929355621337890625e-15;
  y3_239 = y3_238 - _28;
  y2_240 = y2_158 + 1.0e+0;
  goto __BB13;

  __BB(13):
  y2_159 = __PHI (__BB11: y2_158, __BB12: y2_240);
  y3_168 = __PHI (__BB11: y3_238, __BB12: y3_239);
  _29 = xdbl_193 + 8ul;
  __MEM <double> (_29) = y2_159;
  _30 = xdbl_193 + 80ul;
  _31 = __MEM <double> (_30);
  _32 = xdbl_193 + 24ul;
  _33 = __MEM <double> (_32);
  y1_242 = _31 - _33;
  if (y3_168 < 0.0)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  _34 = 3.552713678800500929355621337890625e-15;
  y1_243 = y1_242 - _34;
  y3_244 = y3_168 + 1.0e+0;
  goto __BB15;

  __BB(15):
  y1_151 = __PHI (__BB13: y1_242, __BB14: y1_243);
  y3_169 = __PHI (__BB13: y3_168, __BB14: y3_244);
  _35 = xdbl_193 + 16ul;
  __MEM <double> (_35) = y3_169;
  _36 = xdbl_193 + 88ul;
  _37 = __MEM <double> (_36);
  _38 = xdbl_193 + 32ul;
  _39 = __MEM <double> (_38);
  y2_246 = _37 - _39;
  if (y1_151 < 0.0)
    goto __BB16;
  else
    goto __BB17;

  __BB(16):
  _40 = 3.552713678800500929355621337890625e-15;
  y2_247 = y2_246 - _40;
  y1_248 = y1_151 + 1.0e+0;
  goto __BB17;

  __BB(17):
  y1_152 = __PHI (__BB15: y1_151, __BB16: y1_248);
  y2_160 = __PHI (__BB15: y2_246, __BB16: y2_247);
  _41 = xdbl_193 + 24ul;
  __MEM <double> (_41) = y1_152;
  _42 = __MEM <double> (xdbl_193);
  _43 = xdbl_193 + 40ul;
  _44 = __MEM <double> (_43);
  y3_250 = _42 - _44;
  if (y2_160 < 0.0)
    goto __BB18;
  else
    goto __BB19;

  __BB(18):
  _45 = 3.552713678800500929355621337890625e-15;
  y3_251 = y3_250 - _45;
  y2_252 = y2_160 + 1.0e+0;
  goto __BB19;

  __BB(19):
  y2_161 = __PHI (__BB17: y2_160, __BB18: y2_252);
  y3_170 = __PHI (__BB17: y3_250, __BB18: y3_251);
  _46 = xdbl_193 + 32ul;
  __MEM <double> (_46) = y2_161;
  _47 = xdbl_193 + 8ul;
  _48 = __MEM <double> (_47);
  _49 = xdbl_193 + 48ul;
  _50 = __MEM <double> (_49);
  y1_254 = _48 - _50;
  if (y3_170 < 0.0)
    goto __BB20;
  else
    goto __BB21;

  __BB(20):
  _51 = 3.552713678800500929355621337890625e-15;
  y1_255 = y1_254 - _51;
  y3_256 = y3_170 + 1.0e+0;
  goto __BB21;

  __BB(21):
  y1_153 = __PHI (__BB19: y1_254, __BB20: y1_255);
  y3_171 = __PHI (__BB19: y3_170, __BB20: y3_256);
  _52 = xdbl_193 + 40ul;
  __MEM <double> (_52) = y3_171;
  _53 = xdbl_193 + 16ul;
  _54 = __MEM <double> (_53);
  _55 = xdbl_193 + 56ul;
  _56 = __MEM <double> (_55);
  y2_258 = _54 - _56;
  if (y1_153 < 0.0)
    goto __BB22;
  else
    goto __BB23;

  __BB(22):
  _57 = 3.552713678800500929355621337890625e-15;
  y2_259 = y2_258 - _57;
  y1_260 = y1_153 + 1.0e+0;
  goto __BB23;

  __BB(23):
  y1_154 = __PHI (__BB21: y1_153, __BB22: y1_260);
  y2_162 = __PHI (__BB21: y2_258, __BB22: y2_259);
  _58 = xdbl_193 + 48ul;
  __MEM <double> (_58) = y1_154;
  _59 = xdbl_193 + 24ul;
  _60 = __MEM <double> (_59);
  _61 = xdbl_193 + 64ul;
  _62 = __MEM <double> (_61);
  y3_262 = _60 - _62;
  if (y2_162 < 0.0)
    goto __BB24;
  else
    goto __BB25;

  __BB(24):
  _63 = 3.552713678800500929355621337890625e-15;
  y3_263 = y3_262 - _63;
  y2_264 = y2_162 + 1.0e+0;
  goto __BB25;

  __BB(25):
  y2_163 = __PHI (__BB23: y2_162, __BB24: y2_264);
  y3_172 = __PHI (__BB23: y3_262, __BB24: y3_263);
  _64 = xdbl_193 + 56ul;
  __MEM <double> (_64) = y2_163;
  _65 = xdbl_193 + 32ul;
  _66 = __MEM <double> (_65);
  _67 = xdbl_193 + 72ul;
  _68 = __MEM <double> (_67);
  y1_266 = _66 - _68;
  if (y3_172 < 0.0)
    goto __BB26;
  else
    goto __BB27;

  __BB(26):
  _69 = 3.552713678800500929355621337890625e-15;
  y1_267 = y1_266 - _69;
  y3_268 = y3_172 + 1.0e+0;
  goto __BB27;

  __BB(27):
  y1_155 = __PHI (__BB25: y1_266, __BB26: y1_267);
  y3_173 = __PHI (__BB25: y3_172, __BB26: y3_268);
  _70 = xdbl_193 + 64ul;
  __MEM <double> (_70) = y3_173;
  _71 = xdbl_193 + 40ul;
  _72 = __MEM <double> (_71);
  _73 = xdbl_193 + 80ul;
  _74 = __MEM <double> (_73);
  y2_270 = _72 - _74;
  if (y1_155 < 0.0)
    goto __BB28;
  else
    goto __BB29;

  __BB(28):
  _75 = 3.552713678800500929355621337890625e-15;
  y2_271 = y2_270 - _75;
  y1_272 = y1_155 + 1.0e+0;
  goto __BB29;

  __BB(29):
  y1_156 = __PHI (__BB27: y1_155, __BB28: y1_272);
  y2_164 = __PHI (__BB27: y2_270, __BB28: y2_271);
  _76 = xdbl_193 + 72ul;
  __MEM <double> (_76) = y1_156;
  _77 = xdbl_193 + 48ul;
  _78 = __MEM <double> (_77);
  _79 = xdbl_193 + 88ul;
  _80 = __MEM <double> (_79);
  y3_274 = _78 - _80;
  if (y2_164 < 0.0)
    goto __BB30;
  else
    goto __BB31;

  __BB(30):
  _81 = 3.552713678800500929355621337890625e-15;
  y3_275 = y3_274 - _81;
  y2_276 = y2_164 + 1.0e+0;
  goto __BB31;

  __BB(31):
  y2_165 = __PHI (__BB29: y2_164, __BB30: y2_276);
  y3_174 = __PHI (__BB29: y3_274, __BB30: y3_275);
  _82 = xdbl_193 + 80ul;
  __MEM <double> (_82) = y2_165;
  if (y3_174 < 0.0)
    goto __BB32;
  else
    goto __BB33;

  __BB(32):
  carry_279 = 3.552713678800500929355621337890625e-15;
  y3_280 = y3_174 + 1.0e+0;
  goto __BB34;

  __BB(33):
  carry_278 = 0.0;
  goto __BB34;

  __BB(34):
  y3_175 = __PHI (__BB32: y3_280, __BB33: y3_174);
  carry_178 = __PHI (__BB32: carry_279, __BB33: carry_278);
  _83 = xdbl_193 + 88ul;
  __MEM <double> (_83) = y3_175;
  k_282 = k_145 + 12;
  goto __BB35;

  __BB(35,loop_header(2)):
  k_145 = __PHI (__BB8: k_144, __BB34: k_282);
  carry_179 = __PHI (__BB8: carry_177, __BB34: carry_178);
  if (k_145 <= kmax_200)
    goto __BB9;
  else
    goto __BB36;

  __BB(36):
  _84 = state_191(D)->pr;
  kmax_201 = (int) _84;
  goto __BB41;

  __BB(37):
  _85 = (long unsigned int) jr_185;
  _86 = _85 * 8ul;
  _87 = xdbl_193 + _86;
  _88 = __MEM <double> (_87);
  _89 = (long unsigned int) ir_183;
  _90 = _89 * 8ul;
  _91 = xdbl_193 + _90;
  _92 = __MEM <double> (_91);
  y1_222 = _88 - _92;
  y2_223 = y1_222 - carry_181;
  if (y2_223 < 0.0)
    goto __BB38;
  else
    goto __BB39;

  __BB(38):
  carry_225 = 3.552713678800500929355621337890625e-15;
  y2_226 = y2_223 + 1.0e+0;
  goto __BB40;

  __BB(39):
  carry_224 = 0.0;
  goto __BB40;

  __BB(40):
  y2_166 = __PHI (__BB38: y2_226, __BB39: y2_223);
  carry_180 = __PHI (__BB38: carry_225, __BB39: carry_224);
  _93 = (long unsigned int) ir_183;
  _94 = _93 * 8ul;
  _95 = xdbl_193 + _94;
  __MEM <double> (_95) = y2_166;
  _96 = 2.68435456e+8;
  _97 = (long unsigned int) ir_183;
  _98 = _97 * 8ul;
  _99 = ydbl_194 + _98;
  _100 = y2_166 + _96;
  __MEM <double> (_99) = _100;
  _101 = next[ir_183];
  ir_229 = (unsigned int) _101;
  _102 = next[jr_185];
  jr_230 = (unsigned int) _102;
  k_231 = k_146 + 1;
  goto __BB41;

  __BB(41,loop_header(3)):
  k_146 = __PHI (__BB36: k_145, __BB40: k_231);
  carry_181 = __PHI (__BB36: carry_179, __BB40: carry_180);
  ir_183 = __PHI (__BB36: ir_182, __BB40: ir_229);
  jr_185 = __PHI (__BB36: jr_184, __BB40: jr_230);
  if (k_146 < kmax_201)
    goto __BB37;
  else
    goto __BB42;

  __BB(42):
  _103 = (long unsigned int) ir_183;
  _104 = _103 * 8ul;
  _105 = xdbl_193 + _104;
  _106 = __MEM <double> (_105);
  _107 = 2.68435456e+8;
  _108 = (long unsigned int) ir_183;
  _109 = _108 * 8ul;
  _110 = ydbl_194 + _109;
  _111 = _106 + _107;
  __MEM <double> (_110) = _111;
  k_203 = next[ir_183];
  goto __BB44;

  __BB(43):
  _112 = (long unsigned int) k_147;
  _113 = _112 * 8ul;
  _114 = xdbl_193 + _113;
  _115 = __MEM <double> (_114);
  _116 = 2.68435456e+8;
  _117 = (long unsigned int) k_147;
  _118 = _117 * 8ul;
  _119 = ydbl_194 + _118;
  _120 = _115 + _116;
  __MEM <double> (_119) = _120;
  k_221 = next[k_147];
  goto __BB44;

  __BB(44,loop_header(4)):
  k_147 = __PHI (__BB42: k_203, __BB43: k_221);
  if (k_147 > 0)
    goto __BB43;
  else
    goto __BB45;

  __BB(45):
  k_204 = 0;
  m_205 = 0;
  goto __BB49;

  __BB(46):
  _121 = (long unsigned int) k_148;
  _122 = _121 * 8ul;
  _123 = xdbl_193 + _122;
  x_211 = __MEM <double> (_123);
  _124 = (long unsigned int) k_148;
  _125 = _124 * 8ul;
  _126 = ydbl_194 + _125;
  _127 = __MEM <double> (_126);
  _128 = 2.68435456e+8;
  y2_212 = _127 - _128;
  if (y2_212 > x_211)
    goto __BB47;
  else
    goto __BB48;

  __BB(47):
  _129 = 5.9604644775390625e-8;
  y2_213 = y2_212 - _129;
  goto __BB48;

  __BB(48):
  y2_167 = __PHI (__BB46: y2_212, __BB47: y2_213);
  _130 = x_211 - y2_167;
  _131 = 1.6777216e+7;
  y1_214 = _130 * _131;
  _132 = m_149;
  m_215 = _132 + 1;
  _133 = (long unsigned int) _132;
  _134 = _133 * 4ul;
  _135 = xflt_192 + _134;
  _136 = (float) y1_214;
  __MEM <float> (_135) = _136;
  _137 = m_215;
  m_217 = _137 + 1;
  _138 = (long unsigned int) _137;
  _139 = _138 * 4ul;
  _140 = xflt_192 + _139;
  _141 = (float) y2_167;
  __MEM <float> (_140) = _141;
  k_219 = k_148 + 1;
  goto __BB49;

  __BB(49,loop_header(5)):
  k_148 = __PHI (__BB45: k_204, __BB48: k_219);
  m_149 = __PHI (__BB45: m_205, __BB48: m_217);
  if (k_148 <= 11)
    goto __BB46;
  else
    goto __BB50;

  __BB(50):
  state_191(D)->ir = ir_183;
  _142 = ir_183 * 2u;
  state_191(D)->is = _142;
  _143 = ir_183 * 2u;
  state_191(D)->is_old = _143;
  state_191(D)->jr = jr_185;
  state_191(D)->carry = carry_181;
  return;

}


double __GIMPLE (ssa)
ranlxs_get_double (void * vstate)
{
  const unsigned int is;
  struct ranlxs_state_t * state;
  double D_3243;
  unsigned int _1;
  int _2;
  unsigned int _3;
  unsigned int _4;
  float _5;
  double _13;

  __BB(2):
  state_8 = vstate_7(D);
  _1 = state_8->is;
  _2 = snext[_1];
  is_10 = (const unsigned int) _2;
  state_8->is = is_10;
  _3 = state_8->is_old;
  if (is_10 == _3)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  increment_state (state_8);
  goto __BB4;

  __BB(4):
  _4 = state_8->is;
  _5 = state_8->xflt[_4];
  _13 = (double) _5;
  goto __BB5;

  __BB(5):
L2:
  return _13;

}


long unsigned int __GIMPLE (ssa)
ranlxs_get (void * vstate)
{
  long unsigned int D_3239;
  double _1;
  double _2;
  long unsigned int _6;

  __BB(2):
  _1 = ranlxs_get_double (vstate_4(D));
  _2 = _1 * 1.6777216e+7;
  _6 = (long unsigned int) _2;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
ranlxs_set_lux (void * vstate, long unsigned int s, unsigned int luxury)
{
  long int seed;
  double y;
  double x;
  int xbit[31];
  int m;
  int k;
  int i;
  int jbit;
  int ibit;
  struct ranlxs_state_t * state;
  int _1;
  int _2;
  int _3;
  double _4;
  int _5;
  int _6;
  int _7;
  int _8;
  int _9;
  int _10;
  double _11;
  double _12;

  __BB(2):
  state_27 = vstate_26(D);
  if (s_28(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_29 = 1ul;
  goto __BB4;

  __BB(4):
  s_13 = __PHI (__BB2: s_28(D), __BB3: s_29);
  seed_30 = (long int) s_13;
  _1 = (int) seed_30;
  i_31 = _1 & 2147483647;
  k_32 = 0;
  goto __BB6;

  __BB(5):
  _2 = i_18 % 2;
  xbit[k_19] = _2;
  i_56 = i_18 / 2;
  k_57 = k_19 + 1;
  goto __BB6;

  __BB(6,loop_header(1)):
  i_18 = __PHI (__BB4: i_31, __BB5: i_56);
  k_19 = __PHI (__BB4: k_32, __BB5: k_57);
  if (k_19 <= 30)
    goto __BB5;
  else
    goto __BB7;

  __BB(7):
  ibit_34 = 0;
  jbit_35 = 18;
  k_36 = 0;
  goto __BB12;

  __BB(8):
  x_45 = 0.0;
  m_46 = 1;
  goto __BB10;

  __BB(9):
  _3 = xbit[ibit_14];
  y_49 = (double) _3;
  _4 = x_22 + y_49;
  x_50 = x_22 + _4;
  _5 = xbit[ibit_14];
  _6 = xbit[jbit_16];
  _7 = _5 + _6;
  _8 = _7 % 2;
  xbit[ibit_14] = _8;
  _9 = ibit_14 + 1;
  ibit_52 = _9 % 31;
  _10 = jbit_16 + 1;
  jbit_53 = _10 % 31;
  m_54 = m_21 + 1;
  goto __BB10;

  __BB(10,loop_header(3)):
  ibit_14 = __PHI (__BB8: ibit_15, __BB9: ibit_52);
  jbit_16 = __PHI (__BB8: jbit_17, __BB9: jbit_53);
  m_21 = __PHI (__BB8: m_46, __BB9: m_54);
  x_22 = __PHI (__BB8: x_45, __BB9: x_50);
  if (m_21 <= 48)
    goto __BB9;
  else
    goto __BB11;

  __BB(11):
  _11 = 3.552713678800500929355621337890625e-15;
  _12 = x_22 * _11;
  state_27->xdbl[k_20] = _12;
  k_48 = k_20 + 1;
  goto __BB12;

  __BB(12,loop_header(2)):
  ibit_15 = __PHI (__BB7: ibit_34, __BB11: ibit_14);
  jbit_17 = __PHI (__BB7: jbit_35, __BB11: jbit_16);
  k_20 = __PHI (__BB7: k_36, __BB11: k_48);
  if (k_20 <= 11)
    goto __BB8;
  else
    goto __BB13;

  __BB(13):
  state_27->carry = 0.0;
  state_27->ir = 0u;
  state_27->jr = 7u;
  state_27->is = 23u;
  state_27->is_old = 0u;
  state_27->pr = luxury_42(D);
  xbit ={v} _Literal (int[31]) {CLOBBER};
  return;

}


void __GIMPLE (ssa)
ranlxs0_set (void * vstate, long unsigned int s)
{
  __BB(2):
  ranlxs_set_lux (vstate_2(D), s_3(D), 109u);
  return;

}


void __GIMPLE (ssa)
ranlxs1_set (void * vstate, long unsigned int s)
{
  __BB(2):
  ranlxs_set_lux (vstate_2(D), s_3(D), 202u);
  return;

}


void __GIMPLE (ssa)
ranlxs2_set (void * vstate, long unsigned int s)
{
  __BB(2):
  ranlxs_set_lux (vstate_2(D), s_3(D), 397u);
  return;

}


