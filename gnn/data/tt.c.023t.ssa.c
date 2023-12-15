long unsigned int __GIMPLE (ssa)
tt_get (void * vstate)
{
  int i;
  int n;
  long unsigned int * const x;
  long unsigned int y;
  const long unsigned int mag01[2];
  struct tt_state_t * state;
  long unsigned int D_3130;
  sizetype _1;
  sizetype _2;
  sizetype _3;
  long unsigned int * _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int * _8;
  long unsigned int _9;
  long unsigned int _10;
  long unsigned int _11;
  long unsigned int _12;
  long unsigned int _13;
  long unsigned int * _14;
  long unsigned int _15;
  long unsigned int _16;
  long unsigned int _17;
  long unsigned int _18;
  long unsigned int _19;
  long unsigned int * _20;
  long unsigned int _21;
  long unsigned int _22;
  long unsigned int _23;
  sizetype _24;
  long unsigned int * _25;
  long unsigned int _26;
  long unsigned int _27;
  long unsigned int _28;
  long unsigned int * _29;
  long unsigned int _30;
  long unsigned int _31;
  long unsigned int _32;
  long unsigned int _33;
  long unsigned int _34;
  long unsigned int * _35;
  long unsigned int _36;
  long unsigned int _37;
  long unsigned int _38;
  long unsigned int _39;
  long unsigned int _40;
  long unsigned int * _41;
  long unsigned int _42;
  long unsigned int _43;
  long unsigned int _44;
  long unsigned int * _45;
  long unsigned int _46;
  long unsigned int _47;
  long unsigned int _48;
  long unsigned int _49;
  long unsigned int _50;
  int _51;
  long unsigned int _77;

  __BB(2):
  state_59 = vstate_58(D);
  mag01[0ul] = 0ul;
  mag01[1ul] = 2394935336ul;
  x_63 = &state_59->x;
  n_64 = state_59->n;
  if (n_64 > 24)
    goto __BB3;
  else
    goto __BB10;

  __BB(3):
  i_65 = 0;
  goto __BB5;

  __BB(4):
  _1 = (sizetype) i_53;
  _2 = _1 + 7ul;
  _3 = _2 * 8ul;
  _4 = x_63 + _3;
  _5 = __MEM <long unsigned int> ((long unsigned int * const)_4);
  _6 = (long unsigned int) i_53;
  _7 = _6 * 8ul;
  _8 = x_63 + _7;
  _9 = __MEM <long unsigned int> ((long unsigned int * const)_8);
  _10 = _9 >> 1;
  _11 = _5 ^ _10;
  _12 = (long unsigned int) i_53;
  _13 = _12 * 8ul;
  _14 = x_63 + _13;
  _15 = __MEM <long unsigned int> ((long unsigned int * const)_14);
  _16 = _15 & 1ul;
  _17 = mag01[_16];
  _18 = (long unsigned int) i_53;
  _19 = _18 * 8ul;
  _20 = x_63 + _19;
  _21 = _11 ^ _17;
  __MEM <long unsigned int> ((long unsigned int * const)_20) = _21;
  i_70 = i_53 + 1;
  goto __BB5;

  __BB(5,loop_header(1)):
  i_53 = __PHI (__BB3: i_65, __BB4: i_70);
  if (i_53 <= 17)
    goto __BB4;
  else
    goto __BB6;

  __BB(6):
  goto __BB8;

  __BB(7):
  _22 = (long unsigned int) i_54;
  _23 = _22 * 8ul;
  _24 = _23 + 18446744073709551472ul;
  _25 = x_63 + _24;
  _26 = __MEM <long unsigned int> ((long unsigned int * const)_25);
  _27 = (long unsigned int) i_54;
  _28 = _27 * 8ul;
  _29 = x_63 + _28;
  _30 = __MEM <long unsigned int> ((long unsigned int * const)_29);
  _31 = _30 >> 1;
  _32 = _26 ^ _31;
  _33 = (long unsigned int) i_54;
  _34 = _33 * 8ul;
  _35 = x_63 + _34;
  _36 = __MEM <long unsigned int> ((long unsigned int * const)_35);
  _37 = _36 & 1ul;
  _38 = mag01[_37];
  _39 = (long unsigned int) i_54;
  _40 = _39 * 8ul;
  _41 = x_63 + _40;
  _42 = _32 ^ _38;
  __MEM <long unsigned int> ((long unsigned int * const)_41) = _42;
  i_68 = i_54 + 1;
  goto __BB8;

  __BB(8,loop_header(2)):
  i_54 = __PHI (__BB6: i_53, __BB7: i_68);
  if (i_54 <= 24)
    goto __BB7;
  else
    goto __BB9;

  __BB(9):
  n_66 = 0;
  goto __BB10;

  __BB(10):
  n_52 = __PHI (__BB2: n_64, __BB9: n_66);
  _43 = (long unsigned int) n_52;
  _44 = _43 * 8ul;
  _45 = x_63 + _44;
  y_71 = __MEM <long unsigned int> ((long unsigned int * const)_45);
  _46 = y_71 << 7;
  _47 = _46 & 727393536ul;
  y_72 = y_71 ^ _47;
  _48 = y_72 << 15;
  _49 = _48 & 3683319808ul;
  y_73 = y_72 ^ _49;
  y_74 = y_73 & 4294967295ul;
  _50 = y_74 >> 16;
  y_75 = y_74 ^ _50;
  _51 = n_52 + 1;
  state_59->n = _51;
  _77 = y_75;
  mag01 ={v} _Literal (const long unsigned int[2]) {CLOBBER};
  goto __BB11;

  __BB(11):
L9:
  return _77;

}


double __GIMPLE (ssa)
tt_get_double (void * vstate)
{
  double D_3126;
  long unsigned int _1;
  double _2;
  double _6;

  __BB(2):
  _1 = tt_get (vstate_4(D));
  _2 = (double) _1;
  _6 = _2 / 4.294967296e+9;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


void __GIMPLE (ssa)
tt_set (void * vstate, long unsigned int s)
{
  int i;
  static const struct tt_state_t init_state = _Literal (const struct tt_state_t) {.n=0, .x=_Literal (long unsigned int[25]) {2515684779ul, 191386133ul, 3882666727ul, 2940125753ul, 1902095651ul, 614830253ul, 1776596463ul, 3208995137ul, 2528910203ul, 2814244901ul, 3252581815ul, 2287512009ul, 766015123ul, 3059218909ul, 4292643487ul, 2166479473ul, 2340568779ul, 2287797749ul, 1310772551ul, 1520096729ul, 1361841155ul, 3934616781ul, 1287770895ul, 2291247265ul, 2797054683ul}};
  struct tt_state_t * state;
  long unsigned int _1;
  int _2;
  long unsigned int _3;
  long unsigned int _4;
  long unsigned int _5;

  __BB(2):
  state_10 = vstate_9(D);
  if (s_11(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  state_10->n = 0;
  state_10->x[0ul] = 2515684779ul;
  state_10->x[1ul] = 191386133ul;
  state_10->x[2ul] = 3882666727ul;
  state_10->x[3ul] = 2940125753ul;
  state_10->x[4ul] = 1902095651ul;
  state_10->x[5ul] = 614830253ul;
  state_10->x[6ul] = 1776596463ul;
  state_10->x[7ul] = 3208995137ul;
  state_10->x[8ul] = 2528910203ul;
  state_10->x[9ul] = 2814244901ul;
  state_10->x[10ul] = 3252581815ul;
  state_10->x[11ul] = 2287512009ul;
  state_10->x[12ul] = 766015123ul;
  state_10->x[13ul] = 3059218909ul;
  state_10->x[14ul] = 4292643487ul;
  state_10->x[15ul] = 2166479473ul;
  state_10->x[16ul] = 2340568779ul;
  state_10->x[17ul] = 2287797749ul;
  state_10->x[18ul] = 1310772551ul;
  state_10->x[19ul] = 1520096729ul;
  state_10->x[20ul] = 1361841155ul;
  state_10->x[21ul] = 3934616781ul;
  state_10->x[22ul] = 1287770895ul;
  state_10->x[23ul] = 2291247265ul;
  state_10->x[24ul] = 2797054683ul;
  goto __BB7;

  __BB(4):
  state_10->n = 0;
  _1 = s_11(D) & 4294967295ul;
  state_10->x[0] = _1;
  i_15 = 1;
  goto __BB6;

  __BB(5):
  _2 = i_6 + _Literal (int) -1;
  _3 = state_10->x[_2];
  _4 = _3 * 69069ul;
  _5 = _4 & 4294967295ul;
  state_10->x[i_6] = _5;
  i_17 = i_6 + 1;
  goto __BB6;

  __BB(6,loop_header(1)):
  i_6 = __PHI (__BB4: i_15, __BB5: i_17);
  if (i_6 <= 24)
    goto __BB5;
  else
    goto __BB7;

  __BB(7):
  goto __BB8;

  __BB(8):
L6:
  return;

}


