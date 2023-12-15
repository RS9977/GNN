void __GIMPLE (ssa)
increment_state (struct ranlxd_state_t * state)
{
  unsigned int jr;
  unsigned int ir;
  double carry;
  double * xdbl;
  double y3;
  double y2;
  double y1;
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
  int _96;
  int _97;

  __BB(2):
  xdbl_140 = &state_139(D)->xdbl;
  carry_142 = state_139(D)->carry;
  ir_143 = state_139(D)->ir;
  jr_144 = state_139(D)->jr;
  k_145 = 0;
  goto __BB7;

  __BB(3):
  _1 = (long unsigned int) jr_134;
  _2 = _1 * 8ul;
  _3 = xdbl_140 + _2;
  _4 = __MEM <double> (_3);
  _5 = (long unsigned int) ir_132;
  _6 = _5 * 8ul;
  _7 = xdbl_140 + _6;
  _8 = __MEM <double> (_7);
  y1_212 = _4 - _8;
  y2_213 = y1_212 - carry_127;
  if (y2_213 < 0.0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  carry_215 = 3.552713678800500929355621337890625e-15;
  y2_216 = y2_213 + 1.0e+0;
  goto __BB6;

  __BB(5):
  carry_214 = 0.0;
  goto __BB6;

  __BB(6):
  y2_108 = __PHI (__BB4: y2_216, __BB5: y2_213);
  carry_126 = __PHI (__BB4: carry_215, __BB5: carry_214);
  _9 = (long unsigned int) ir_132;
  _10 = _9 * 8ul;
  _11 = xdbl_140 + _10;
  __MEM <double> (_11) = y2_108;
  _12 = next[ir_132];
  ir_218 = (unsigned int) _12;
  _13 = next[jr_134];
  jr_219 = (unsigned int) _13;
  k_220 = k_98 + 1;
  goto __BB7;

  __BB(7,loop_header(1)):
  k_98 = __PHI (__BB2: k_145, __BB6: k_220);
  carry_127 = __PHI (__BB2: carry_142, __BB6: carry_126);
  ir_132 = __PHI (__BB2: ir_143, __BB6: ir_218);
  jr_134 = __PHI (__BB2: jr_144, __BB6: jr_219);
  if (ir_132 != 0u)
    goto __BB3;
  else
    goto __BB8;

  __BB(8):
  _14 = state_139(D)->pr;
  _15 = _14 + 4294967284u;
  kmax_146 = (int) _15;
  goto __BB35;

  __BB(9):
  _16 = xdbl_140 + 56ul;
  _17 = __MEM <double> (_16);
  _18 = __MEM <double> (xdbl_140);
  y1_161 = _17 - _18;
  y1_162 = y1_161 - carry_129;
  _19 = xdbl_140 + 64ul;
  _20 = __MEM <double> (_19);
  _21 = xdbl_140 + 8ul;
  _22 = __MEM <double> (_21);
  y2_163 = _20 - _22;
  if (y1_162 < 0.0)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  _23 = 3.552713678800500929355621337890625e-15;
  y2_164 = y2_163 - _23;
  y1_165 = y1_162 + 1.0e+0;
  goto __BB11;

  __BB(11):
  y1_101 = __PHI (__BB9: y1_162, __BB10: y1_165);
  y2_109 = __PHI (__BB9: y2_163, __BB10: y2_164);
  __MEM <double> (xdbl_140) = y1_101;
  _24 = xdbl_140 + 72ul;
  _25 = __MEM <double> (_24);
  _26 = xdbl_140 + 16ul;
  _27 = __MEM <double> (_26);
  y3_167 = _25 - _27;
  if (y2_109 < 0.0)
    goto __BB12;
  else
    goto __BB13;

  __BB(12):
  _28 = 3.552713678800500929355621337890625e-15;
  y3_168 = y3_167 - _28;
  y2_169 = y2_109 + 1.0e+0;
  goto __BB13;

  __BB(13):
  y2_110 = __PHI (__BB11: y2_109, __BB12: y2_169);
  y3_118 = __PHI (__BB11: y3_167, __BB12: y3_168);
  _29 = xdbl_140 + 8ul;
  __MEM <double> (_29) = y2_110;
  _30 = xdbl_140 + 80ul;
  _31 = __MEM <double> (_30);
  _32 = xdbl_140 + 24ul;
  _33 = __MEM <double> (_32);
  y1_171 = _31 - _33;
  if (y3_118 < 0.0)
    goto __BB14;
  else
    goto __BB15;

  __BB(14):
  _34 = 3.552713678800500929355621337890625e-15;
  y1_172 = y1_171 - _34;
  y3_173 = y3_118 + 1.0e+0;
  goto __BB15;

  __BB(15):
  y1_102 = __PHI (__BB13: y1_171, __BB14: y1_172);
  y3_119 = __PHI (__BB13: y3_118, __BB14: y3_173);
  _35 = xdbl_140 + 16ul;
  __MEM <double> (_35) = y3_119;
  _36 = xdbl_140 + 88ul;
  _37 = __MEM <double> (_36);
  _38 = xdbl_140 + 32ul;
  _39 = __MEM <double> (_38);
  y2_175 = _37 - _39;
  if (y1_102 < 0.0)
    goto __BB16;
  else
    goto __BB17;

  __BB(16):
  _40 = 3.552713678800500929355621337890625e-15;
  y2_176 = y2_175 - _40;
  y1_177 = y1_102 + 1.0e+0;
  goto __BB17;

  __BB(17):
  y1_103 = __PHI (__BB15: y1_102, __BB16: y1_177);
  y2_111 = __PHI (__BB15: y2_175, __BB16: y2_176);
  _41 = xdbl_140 + 24ul;
  __MEM <double> (_41) = y1_103;
  _42 = __MEM <double> (xdbl_140);
  _43 = xdbl_140 + 40ul;
  _44 = __MEM <double> (_43);
  y3_179 = _42 - _44;
  if (y2_111 < 0.0)
    goto __BB18;
  else
    goto __BB19;

  __BB(18):
  _45 = 3.552713678800500929355621337890625e-15;
  y3_180 = y3_179 - _45;
  y2_181 = y2_111 + 1.0e+0;
  goto __BB19;

  __BB(19):
  y2_112 = __PHI (__BB17: y2_111, __BB18: y2_181);
  y3_120 = __PHI (__BB17: y3_179, __BB18: y3_180);
  _46 = xdbl_140 + 32ul;
  __MEM <double> (_46) = y2_112;
  _47 = xdbl_140 + 8ul;
  _48 = __MEM <double> (_47);
  _49 = xdbl_140 + 48ul;
  _50 = __MEM <double> (_49);
  y1_183 = _48 - _50;
  if (y3_120 < 0.0)
    goto __BB20;
  else
    goto __BB21;

  __BB(20):
  _51 = 3.552713678800500929355621337890625e-15;
  y1_184 = y1_183 - _51;
  y3_185 = y3_120 + 1.0e+0;
  goto __BB21;

  __BB(21):
  y1_104 = __PHI (__BB19: y1_183, __BB20: y1_184);
  y3_121 = __PHI (__BB19: y3_120, __BB20: y3_185);
  _52 = xdbl_140 + 40ul;
  __MEM <double> (_52) = y3_121;
  _53 = xdbl_140 + 16ul;
  _54 = __MEM <double> (_53);
  _55 = xdbl_140 + 56ul;
  _56 = __MEM <double> (_55);
  y2_187 = _54 - _56;
  if (y1_104 < 0.0)
    goto __BB22;
  else
    goto __BB23;

  __BB(22):
  _57 = 3.552713678800500929355621337890625e-15;
  y2_188 = y2_187 - _57;
  y1_189 = y1_104 + 1.0e+0;
  goto __BB23;

  __BB(23):
  y1_105 = __PHI (__BB21: y1_104, __BB22: y1_189);
  y2_113 = __PHI (__BB21: y2_187, __BB22: y2_188);
  _58 = xdbl_140 + 48ul;
  __MEM <double> (_58) = y1_105;
  _59 = xdbl_140 + 24ul;
  _60 = __MEM <double> (_59);
  _61 = xdbl_140 + 64ul;
  _62 = __MEM <double> (_61);
  y3_191 = _60 - _62;
  if (y2_113 < 0.0)
    goto __BB24;
  else
    goto __BB25;

  __BB(24):
  _63 = 3.552713678800500929355621337890625e-15;
  y3_192 = y3_191 - _63;
  y2_193 = y2_113 + 1.0e+0;
  goto __BB25;

  __BB(25):
  y2_114 = __PHI (__BB23: y2_113, __BB24: y2_193);
  y3_122 = __PHI (__BB23: y3_191, __BB24: y3_192);
  _64 = xdbl_140 + 56ul;
  __MEM <double> (_64) = y2_114;
  _65 = xdbl_140 + 32ul;
  _66 = __MEM <double> (_65);
  _67 = xdbl_140 + 72ul;
  _68 = __MEM <double> (_67);
  y1_195 = _66 - _68;
  if (y3_122 < 0.0)
    goto __BB26;
  else
    goto __BB27;

  __BB(26):
  _69 = 3.552713678800500929355621337890625e-15;
  y1_196 = y1_195 - _69;
  y3_197 = y3_122 + 1.0e+0;
  goto __BB27;

  __BB(27):
  y1_106 = __PHI (__BB25: y1_195, __BB26: y1_196);
  y3_123 = __PHI (__BB25: y3_122, __BB26: y3_197);
  _70 = xdbl_140 + 64ul;
  __MEM <double> (_70) = y3_123;
  _71 = xdbl_140 + 40ul;
  _72 = __MEM <double> (_71);
  _73 = xdbl_140 + 80ul;
  _74 = __MEM <double> (_73);
  y2_199 = _72 - _74;
  if (y1_106 < 0.0)
    goto __BB28;
  else
    goto __BB29;

  __BB(28):
  _75 = 3.552713678800500929355621337890625e-15;
  y2_200 = y2_199 - _75;
  y1_201 = y1_106 + 1.0e+0;
  goto __BB29;

  __BB(29):
  y1_107 = __PHI (__BB27: y1_106, __BB28: y1_201);
  y2_115 = __PHI (__BB27: y2_199, __BB28: y2_200);
  _76 = xdbl_140 + 72ul;
  __MEM <double> (_76) = y1_107;
  _77 = xdbl_140 + 48ul;
  _78 = __MEM <double> (_77);
  _79 = xdbl_140 + 88ul;
  _80 = __MEM <double> (_79);
  y3_203 = _78 - _80;
  if (y2_115 < 0.0)
    goto __BB30;
  else
    goto __BB31;

  __BB(30):
  _81 = 3.552713678800500929355621337890625e-15;
  y3_204 = y3_203 - _81;
  y2_205 = y2_115 + 1.0e+0;
  goto __BB31;

  __BB(31):
  y2_116 = __PHI (__BB29: y2_115, __BB30: y2_205);
  y3_124 = __PHI (__BB29: y3_203, __BB30: y3_204);
  _82 = xdbl_140 + 80ul;
  __MEM <double> (_82) = y2_116;
  if (y3_124 < 0.0)
    goto __BB32;
  else
    goto __BB33;

  __BB(32):
  carry_208 = 3.552713678800500929355621337890625e-15;
  y3_209 = y3_124 + 1.0e+0;
  goto __BB34;

  __BB(33):
  carry_207 = 0.0;
  goto __BB34;

  __BB(34):
  y3_125 = __PHI (__BB32: y3_209, __BB33: y3_124);
  carry_128 = __PHI (__BB32: carry_208, __BB33: carry_207);
  _83 = xdbl_140 + 88ul;
  __MEM <double> (_83) = y3_125;
  k_211 = k_99 + 12;
  goto __BB35;

  __BB(35,loop_header(2)):
  k_99 = __PHI (__BB8: k_98, __BB34: k_211);
  carry_129 = __PHI (__BB8: carry_127, __BB34: carry_128);
  if (k_99 <= kmax_146)
    goto __BB9;
  else
    goto __BB36;

  __BB(36):
  _84 = state_139(D)->pr;
  kmax_147 = (int) _84;
  goto __BB41;

  __BB(37):
  _85 = (long unsigned int) jr_135;
  _86 = _85 * 8ul;
  _87 = xdbl_140 + _86;
  _88 = __MEM <double> (_87);
  _89 = (long unsigned int) ir_133;
  _90 = _89 * 8ul;
  _91 = xdbl_140 + _90;
  _92 = __MEM <double> (_91);
  y1_152 = _88 - _92;
  y2_153 = y1_152 - carry_131;
  if (y2_153 < 0.0)
    goto __BB38;
  else
    goto __BB39;

  __BB(38):
  carry_155 = 3.552713678800500929355621337890625e-15;
  y2_156 = y2_153 + 1.0e+0;
  goto __BB40;

  __BB(39):
  carry_154 = 0.0;
  goto __BB40;

  __BB(40):
  y2_117 = __PHI (__BB38: y2_156, __BB39: y2_153);
  carry_130 = __PHI (__BB38: carry_155, __BB39: carry_154);
  _93 = (long unsigned int) ir_133;
  _94 = _93 * 8ul;
  _95 = xdbl_140 + _94;
  __MEM <double> (_95) = y2_117;
  _96 = next[ir_133];
  ir_158 = (unsigned int) _96;
  _97 = next[jr_135];
  jr_159 = (unsigned int) _97;
  k_160 = k_100 + 1;
  goto __BB41;

  __BB(41,loop_header(3)):
  k_100 = __PHI (__BB36: k_99, __BB40: k_160);
  carry_131 = __PHI (__BB36: carry_129, __BB40: carry_130);
  ir_133 = __PHI (__BB36: ir_132, __BB40: ir_158);
  jr_135 = __PHI (__BB36: jr_134, __BB40: jr_159);
  if (k_100 < kmax_147)
    goto __BB37;
  else
    goto __BB42;

  __BB(42):
  state_139(D)->ir = ir_133;
  state_139(D)->ir_old = ir_133;
  state_139(D)->jr = jr_135;
  state_139(D)->carry = carry_131;
  return;

}


double __GIMPLE (ssa)
ranlxd_get_double (void * vstate)
{
  int ir;
  struct ranlxd_state_t * state;
  double D_3181;
  unsigned int _1;
  int _2;
  unsigned int _3;
  unsigned int _4;
  unsigned int _5;
  unsigned int _6;
  double _14;

  __BB(2):
  state_9 = vstate_8(D);
  _1 = state_9->ir;
  ir_11 = (int) _1;
  _2 = next[ir_11];
  _3 = (unsigned int) _2;
  state_9->ir = _3;
  _4 = state_9->ir;
  _5 = state_9->ir_old;
  if (_4 == _5)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  increment_state (state_9);
  goto __BB4;

  __BB(4):
  _6 = state_9->ir;
  _14 = state_9->xdbl[_6];
  goto __BB5;

  __BB(5):
L2:
  return _14;

}


long unsigned int __GIMPLE (ssa)
ranlxd_get (void * vstate)
{
  long unsigned int D_3216;
  double _1;
  double _2;
  long unsigned int _6;

  __BB(2):
  _1 = ranlxd_get_double (vstate_4(D));
  _2 = _1 * 4.294967296e+9;
  _6 = (long unsigned int) _2;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
ranlxd_set_lux (void * vstate, long unsigned int s, unsigned int luxury)
{
  long int seed;
  double y;
  double x;
  int xbit[31];
  int l;
  int k;
  int i;
  int jbit;
  int ibit;
  struct ranlxd_state_t * state;
  int _1;
  int _2;
  int _3;
  int _4;
  double _5;
  int _6;
  int _7;
  int _8;
  int _9;
  int _10;
  int _11;
  double _12;
  double _13;

  __BB(2):
  state_28 = vstate_27(D);
  if (s_29(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  s_30 = 1ul;
  goto __BB4;

  __BB(4):
  s_14 = __PHI (__BB2: s_29(D), __BB3: s_30);
  seed_31 = (long int) s_14;
  i_32 = (int) seed_31;
  k_33 = 0;
  goto __BB6;

  __BB(5):
  _1 = i_19 % 2;
  xbit[k_20] = _1;
  i_56 = i_19 / 2;
  k_57 = k_20 + 1;
  goto __BB6;

  __BB(6,loop_header(1)):
  i_19 = __PHI (__BB4: i_32, __BB5: i_56);
  k_20 = __PHI (__BB4: k_33, __BB5: k_57);
  if (k_20 <= 30)
    goto __BB5;
  else
    goto __BB7;

  __BB(7):
  ibit_35 = 0;
  jbit_36 = 18;
  k_37 = 0;
  goto __BB12;

  __BB(8):
  x_45 = 0.0;
  l_46 = 1;
  goto __BB10;

  __BB(9):
  _2 = xbit[ibit_15];
  _3 = _2 + 1;
  _4 = _3 % 2;
  y_49 = (double) _4;
  _5 = x_23 + y_49;
  x_50 = x_23 + _5;
  _6 = xbit[ibit_15];
  _7 = xbit[jbit_17];
  _8 = _6 + _7;
  _9 = _8 % 2;
  xbit[ibit_15] = _9;
  _10 = ibit_15 + 1;
  ibit_52 = _10 % 31;
  _11 = jbit_17 + 1;
  jbit_53 = _11 % 31;
  l_54 = l_22 + 1;
  goto __BB10;

  __BB(10,loop_header(3)):
  ibit_15 = __PHI (__BB8: ibit_16, __BB9: ibit_52);
  jbit_17 = __PHI (__BB8: jbit_18, __BB9: jbit_53);
  l_22 = __PHI (__BB8: l_46, __BB9: l_54);
  x_23 = __PHI (__BB8: x_45, __BB9: x_50);
  if (l_22 <= 48)
    goto __BB9;
  else
    goto __BB11;

  __BB(11):
  _12 = 3.552713678800500929355621337890625e-15;
  _13 = x_23 * _12;
  state_28->xdbl[k_21] = _13;
  k_48 = k_21 + 1;
  goto __BB12;

  __BB(12,loop_header(2)):
  ibit_16 = __PHI (__BB7: ibit_35, __BB11: ibit_15);
  jbit_18 = __PHI (__BB7: jbit_36, __BB11: jbit_17);
  k_21 = __PHI (__BB7: k_37, __BB11: k_48);
  if (k_21 <= 11)
    goto __BB8;
  else
    goto __BB13;

  __BB(13):
  state_28->carry = 0.0;
  state_28->ir = 11u;
  state_28->jr = 7u;
  state_28->ir_old = 0u;
  state_28->pr = luxury_42(D);
  xbit ={v} _Literal (int[31]) {CLOBBER};
  return;

}


void __GIMPLE (ssa)
ranlxd1_set (void * vstate, long unsigned int s)
{
  __BB(2):
  ranlxd_set_lux (vstate_2(D), s_3(D), 202u);
  return;

}


void __GIMPLE (ssa)
ranlxd2_set (void * vstate, long unsigned int s)
{
  __BB(2):
  ranlxd_set_lux (vstate_2(D), s_3(D), 397u);
  return;

}


