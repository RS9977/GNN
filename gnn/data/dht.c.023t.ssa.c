struct gsl_dht * __GIMPLE (ssa)
gsl_dht_alloc (size_t size)
{
  struct gsl_dht * t;
  struct gsl_dht * D_4630;
  long unsigned int _1;
  long unsigned int _2;
  void * _3;
  double * _4;
  long unsigned int _5;
  long unsigned int _6;
  long unsigned int _7;
  long unsigned int _8;
  void * _9;
  double * _10;
  double * _11;
  long unsigned int _12;
  long unsigned int _13;
  void * _14;
  double * _15;
  double * _16;
  double * _17;
  struct gsl_dht * _18;
  struct gsl_dht * _33;
  struct gsl_dht * _38;
  struct gsl_dht * _42;
  struct gsl_dht * _45;
  struct gsl_dht * _47;
  struct gsl_dht * _49;

  __BB(2):
  if (size_20(D) == 0ul)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("size == 0", ".//dht.c", 37, 1);
  _49 = _Literal (struct gsl_dht *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(4):
  t_23 = malloc (56ul);
  if (t_23 == _Literal (struct gsl_dht *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("out of memory", ".//dht.c", 43, 8);
  _47 = _Literal (struct gsl_dht *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(6):
  t_23->size = size_20(D);
  t_23->xmax = -1.0e+0;
  t_23->nu = -1.0e+0;
  _1 = size_20(D) + 2ul;
  _2 = _1 * 8ul;
  _3 = malloc (_2);
  t_23->j = _3;
  _4 = t_23->j;
  if (_4 == _Literal (double *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  free (t_23);
  gsl_error ("could not allocate memory for j", ".//dht.c", 55, 8);
  _45 = _Literal (struct gsl_dht *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(8):
  _5 = size_20(D) + 1ul;
  _6 = size_20(D) * _5;
  _7 = _6 / 2ul;
  _8 = _7 * 8ul;
  _9 = malloc (_8);
  t_23->Jjj = _9;
  _10 = t_23->Jjj;
  if (_10 == _Literal (double *) 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  _11 = t_23->j;
  free (_11);
  free (t_23);
  gsl_error ("could not allocate memory for Jjj", ".//dht.c", 63, 8);
  _42 = _Literal (struct gsl_dht *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(10):
  _12 = size_20(D) + 1ul;
  _13 = _12 * 8ul;
  _14 = malloc (_13);
  t_23->J2 = _14;
  _15 = t_23->J2;
  if (_15 == _Literal (double *) 0)
    goto __BB11;
  else
    goto __BB12;

  __BB(11):
  _16 = t_23->Jjj;
  free (_16);
  _17 = t_23->j;
  free (_17);
  free (t_23);
  gsl_error ("could not allocate memory for J2", ".//dht.c", 72, 8);
  _38 = _Literal (struct gsl_dht *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB13;

  __BB(12):
  _33 = t_23;
  goto __BB13;

  __BB(13):
  _18 = __PHI (__BB3: _49, __BB5: _47, __BB7: _45, __BB9: _42, __BB11: _38, __BB12: _33);
L10:
  return _18;

}


int __GIMPLE (ssa)
dht_bessel_zeros (struct gsl_dht * t)
{
  int stat_z;
  struct gsl_sf_result z;
  unsigned int s;
  int D_4643;
  int D_4640;
  double * _1;
  double _2;
  double * _3;
  long unsigned int _4;
  long unsigned int _5;
  double * _6;
  double _7;
  long unsigned int _8;
  long unsigned int _9;
  long unsigned int _10;
  int _13;
  int _21;
  int _23;
  int _26;

  __BB(2):
  stat_z_16 = 0;
  _1 = t_18(D)->j;
  __MEM <double> (_1) = 0.0;
  s_20 = 1u;
  goto __BB4;

  __BB(3):
  _2 = t_18(D)->nu;
  _26 = gsl_sf_bessel_zero_Jnu_e (_2, s_11, &z);
  stat_z_27 = _26 + stat_z_12;
  _3 = t_18(D)->j;
  _4 = (long unsigned int) s_11;
  _5 = _4 * 8ul;
  _6 = _3 + _5;
  _7 = z.val;
  __MEM <double> (_6) = _7;
  s_29 = s_11 + 1u;
  goto __BB4;

  __BB(4,loop_header(1)):
  s_11 = __PHI (__BB2: s_20, __BB3: s_29);
  stat_z_12 = __PHI (__BB2: stat_z_16, __BB3: stat_z_27);
  _8 = (long unsigned int) s_11;
  _9 = t_18(D)->size;
  _10 = _9 + 2ul;
  if (_8 < _10)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  if (stat_z_12 != 0)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  gsl_error ("could not compute bessel zeroes", ".//dht.c", 91, 5);
  _23 = 5;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB8;

  __BB(7):
  _21 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB8;

  __BB(8):
  _13 = __PHI (__BB6: _23, __BB7: _21);
  z ={v} _Literal (struct gsl_sf_result) {CLOBBER};
  goto __BB9;

  __BB(9):
L6:
  return _13;

}


int __GIMPLE (ssa)
gsl_dht_init (struct gsl_dht * t, double nu, double xmax)
{
  struct gsl_sf_result J;
  double arg;
  struct gsl_sf_result J;
  double jN;
  int stat_J;
  int stat_bz;
  size_t m;
  size_t n;
  int D_4660;
  int D_4659;
  int D_4654;
  double _1;
  double * _2;
  long unsigned int _3;
  sizetype _4;
  sizetype _5;
  double * _6;
  double _7;
  double * _8;
  double * _9;
  long unsigned int _10;
  double * _11;
  double _12;
  double _13;
  double _14;
  double _15;
  double * _16;
  long unsigned int _17;
  double * _18;
  double _19;
  long unsigned int _20;
  long unsigned int _21;
  double * _22;
  long unsigned int _23;
  double * _24;
  double _25;
  double * _26;
  long unsigned int _27;
  double * _28;
  double _29;
  double _30;
  double * _31;
  long unsigned int _32;
  long unsigned int _33;
  long unsigned int _34;
  long unsigned int _35;
  long unsigned int _36;
  sizetype _37;
  double * _38;
  double _39;
  long unsigned int _40;
  long unsigned int _41;
  int _49;
  int _70;
  int _72;
  int _77;
  int _83;
  int _89;
  int _91;

  __BB(2):
  if (xmax_55(D) <= 0.0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("xmax is not positive", ".//dht.c", 120, 1);
  _91 = 1;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB20;

  __BB(4):
  if (nu_56(D) < 0.0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("nu is negative", ".//dht.c", 122, 1);
  _89 = 1;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB20;

  __BB(6):
  stat_bz_57 = 0;
  stat_J_58 = 0;
  _1 = t_60(D)->nu;
  if (nu_56(D) != _1)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  t_60(D)->nu = nu_56(D);
  stat_bz_63 = dht_bessel_zeros (t_60(D));
  goto __BB8;

  __BB(8):
  stat_bz_45 = __PHI (__BB6: stat_bz_57, __BB7: stat_bz_63);
  _2 = t_60(D)->j;
  _3 = t_60(D)->size;
  _4 = _3 + 1ul;
  _5 = _4 * 8ul;
  _6 = _2 + _5;
  jN_64 = __MEM <double> (_6);
  t_60(D)->xmax = xmax_55(D);
  _7 = jN_64 / xmax_55(D);
  t_60(D)->kmax = _7;
  _8 = t_60(D)->J2;
  __MEM <double> (_8) = 0.0;
  m_68 = 1ul;
  goto __BB10;

  __BB(9):
  _9 = t_60(D)->j;
  _10 = m_43 * 8ul;
  _11 = _9 + _10;
  _12 = __MEM <double> (_11);
  _13 = nu_56(D) + 1.0e+0;
  _83 = gsl_sf_bessel_Jnu_e (_13, _12, &J);
  stat_J_84 = _83 + stat_J_46;
  _14 = J.val;
  _15 = J.val;
  _16 = t_60(D)->J2;
  _17 = m_43 * 8ul;
  _18 = _16 + _17;
  _19 = _14 * _15;
  __MEM <double> (_18) = _19;
  J ={v} _Literal (struct gsl_sf_result) {CLOBBER};
  m_87 = m_43 + 1ul;
  goto __BB10;

  __BB(10,loop_header(1)):
  m_43 = __PHI (__BB8: m_68, __BB9: m_87);
  stat_J_46 = __PHI (__BB8: stat_J_58, __BB9: stat_J_84);
  _20 = t_60(D)->size;
  _21 = _20 + 1ul;
  if (m_43 < _21)
    goto __BB9;
  else
    goto __BB11;

  __BB(11):
  n_69 = 1ul;
  goto __BB16;

  __BB(12):
  m_73 = 1ul;
  goto __BB14;

  __BB(13):
  _22 = t_60(D)->j;
  _23 = n_42 * 8ul;
  _24 = _22 + _23;
  _25 = __MEM <double> (_24);
  _26 = t_60(D)->j;
  _27 = m_44 * 8ul;
  _28 = _26 + _27;
  _29 = __MEM <double> (_28);
  _30 = _25 * _29;
  arg_75 = _30 / jN_64;
  _77 = gsl_sf_bessel_Jnu_e (nu_56(D), arg_75, &J);
  stat_J_78 = _77 + stat_J_47;
  _31 = t_60(D)->Jjj;
  _32 = n_42 + 18446744073709551615ul;
  _33 = n_42 * _32;
  _34 = _33 / 2ul;
  _35 = m_44 + _34;
  _36 = _35 * 8ul;
  _37 = _36 + 18446744073709551608ul;
  _38 = _31 + _37;
  _39 = J.val;
  __MEM <double> (_38) = _39;
  J ={v} _Literal (struct gsl_sf_result) {CLOBBER};
  m_81 = m_44 + 1ul;
  goto __BB14;

  __BB(14,loop_header(3)):
  m_44 = __PHI (__BB12: m_73, __BB13: m_81);
  stat_J_47 = __PHI (__BB12: stat_J_48, __BB13: stat_J_78);
  if (m_44 <= n_42)
    goto __BB13;
  else
    goto __BB15;

  __BB(15):
  n_74 = n_42 + 1ul;
  goto __BB16;

  __BB(16,loop_header(2)):
  n_42 = __PHI (__BB11: n_69, __BB15: n_74);
  stat_J_48 = __PHI (__BB11: stat_J_46, __BB15: stat_J_47);
  _40 = t_60(D)->size;
  _41 = _40 + 1ul;
  if (n_42 < _41)
    goto __BB12;
  else
    goto __BB17;

  __BB(17):
  if (stat_J_48 != 0)
    goto __BB18;
  else
    goto __BB19;

  __BB(18):
  gsl_error ("error computing bessel function", ".//dht.c", 160, 5);
  _72 = 5;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB20;

  __BB(19):
  _70 = stat_bz_45;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB20;

  __BB(20):
  _49 = __PHI (__BB3: _91, __BB5: _89, __BB18: _72, __BB19: _70);
L17:
  return _49;

}


struct gsl_dht * __GIMPLE (ssa)
gsl_dht_new (size_t size, double nu, double xmax)
{
  struct gsl_dht * dht;
  int status;
  struct gsl_dht * D_4648;
  struct gsl_dht * _1;
  struct gsl_dht * _11;
  struct gsl_dht * _12;
  struct gsl_dht * _13;

  __BB(2):
  dht_6 = gsl_dht_alloc (size_4(D));
  if (dht_6 == _Literal (struct gsl_dht *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _13 = _Literal (struct gsl_dht *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(4):
  status_10 = gsl_dht_init (dht_6, nu_7(D), xmax_8(D));
  if (status_10 != 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _12 = _Literal (struct gsl_dht *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(6):
  _11 = dht_6;
  goto __BB7;

  __BB(7):
  _1 = __PHI (__BB3: _13, __BB5: _12, __BB6: _11);
L4:
  return _1;

}


double __GIMPLE (ssa)
gsl_dht_x_sample (const struct gsl_dht * t, int n)
{
  double D_4664;
  double * _1;
  sizetype _2;
  sizetype _3;
  sizetype _4;
  double * _5;
  double _6;
  double * _7;
  long unsigned int _8;
  sizetype _9;
  sizetype _10;
  double * _11;
  double _12;
  double _13;
  double _14;
  double _18;

  __BB(2):
  _1 = t_16(D)->j;
  _2 = (sizetype) n_17(D);
  _3 = _2 + 1ul;
  _4 = _3 * 8ul;
  _5 = _1 + _4;
  _6 = __MEM <double> ((double * const)_5);
  _7 = t_16(D)->j;
  _8 = t_16(D)->size;
  _9 = _8 + 1ul;
  _10 = _9 * 8ul;
  _11 = _7 + _10;
  _12 = __MEM <double> ((double * const)_11);
  _13 = _6 / _12;
  _14 = t_16(D)->xmax;
  _18 = _13 * _14;
  goto __BB3;

  __BB(3):
L0:
  return _18;

}


double __GIMPLE (ssa)
gsl_dht_k_sample (const struct gsl_dht * t, int n)
{
  double D_4666;
  double * _1;
  sizetype _2;
  sizetype _3;
  sizetype _4;
  double * _5;
  double _6;
  double _7;
  double _11;

  __BB(2):
  _1 = t_9(D)->j;
  _2 = (sizetype) n_10(D);
  _3 = _2 + 1ul;
  _4 = _3 * 8ul;
  _5 = _1 + _4;
  _6 = __MEM <double> ((double * const)_5);
  _7 = t_9(D)->xmax;
  _11 = _6 / _7;
  goto __BB3;

  __BB(3):
L0:
  return _11;

}


void __GIMPLE (ssa)
gsl_dht_free (struct gsl_dht * t)
{
  double * _1;
  double * _2;
  double * _3;

  __BB(2):
  if (t_5(D) == _Literal (struct gsl_dht *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4):
  _1 = t_5(D)->J2;
  free (_1);
  _2 = t_5(D)->Jjj;
  free (_2);
  _3 = t_5(D)->j;
  free (_3);
  free (t_5(D));
  goto __BB5;

  __BB(5):
L2:
  return;

}


int __GIMPLE (ssa)
gsl_dht_apply (const struct gsl_dht * t, double * f_in, double * f_out)
{
  size_t n_local;
  size_t m_local;
  double Y;
  double sum;
  size_t i;
  size_t m;
  const double r;
  const double jN;
  int D_4674;
  double * _1;
  long unsigned int _2;
  sizetype _3;
  sizetype _4;
  double * _5;
  double _6;
  double * _7;
  long unsigned int _8;
  long unsigned int _9;
  long unsigned int _10;
  long unsigned int _11;
  long unsigned int _12;
  double * _13;
  double _14;
  double * _15;
  sizetype _16;
  sizetype _17;
  double * _18;
  double _19;
  long unsigned int _20;
  double * _21;
  double _22;
  double _23;
  long unsigned int _24;
  double _25;
  double _26;
  long unsigned int _27;
  double * _28;
  double _29;
  long unsigned int _30;
  int _42;

  __BB(2):
  _1 = t_38(D)->j;
  _2 = t_38(D)->size;
  _3 = _2 + 1ul;
  _4 = _3 * 8ul;
  _5 = _1 + _4;
  jN_39 = __MEM <double> ((double * const)_5);
  _6 = t_38(D)->xmax;
  r_40 = _6 / jN_39;
  m_41 = 0ul;
  goto __BB10;

  __BB(3):
  sum_43 = 0.0;
  i_44 = 0ul;
  goto __BB8;

  __BB(4):
  if (i_32 < m_31)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  m_local_50 = i_32;
  n_local_51 = m_31;
  goto __BB7;

  __BB(6):
  m_local_48 = m_31;
  n_local_49 = i_32;
  goto __BB7;

  __BB(7):
  m_local_34 = __PHI (__BB5: m_local_50, __BB6: m_local_48);
  n_local_35 = __PHI (__BB5: n_local_51, __BB6: n_local_49);
  _7 = t_38(D)->Jjj;
  _8 = n_local_35 + 1ul;
  _9 = n_local_35 * _8;
  _10 = _9 / 2ul;
  _11 = m_local_34 + _10;
  _12 = _11 * 8ul;
  _13 = _7 + _12;
  _14 = __MEM <double> ((double * const)_13);
  _15 = t_38(D)->J2;
  _16 = i_32 + 1ul;
  _17 = _16 * 8ul;
  _18 = _15 + _17;
  _19 = __MEM <double> ((double * const)_18);
  Y_52 = _14 / _19;
  _20 = i_32 * 8ul;
  _21 = f_in_53(D) + _20;
  _22 = __MEM <double> (_21);
  _23 = Y_52 * _22;
  sum_54 = sum_33 + _23;
  i_55 = i_32 + 1ul;
  goto __BB8;

  __BB(8,loop_header(2)):
  i_32 = __PHI (__BB3: i_44, __BB7: i_55);
  sum_33 = __PHI (__BB3: sum_43, __BB7: sum_54);
  _24 = t_38(D)->size;
  if (i_32 < _24)
    goto __BB4;
  else
    goto __BB9;

  __BB(9):
  _25 = sum_33 * 2.0e+0;
  _26 = r_40 * _25;
  _27 = m_31 * 8ul;
  _28 = f_out_45(D) + _27;
  _29 = r_40 * _26;
  __MEM <double> (_28) = _29;
  m_47 = m_31 + 1ul;
  goto __BB10;

  __BB(10,loop_header(1)):
  m_31 = __PHI (__BB2: m_41, __BB9: m_47);
  _30 = t_38(D)->size;
  if (m_31 < _30)
    goto __BB3;
  else
    goto __BB11;

  __BB(11):
  _42 = 0;
  goto __BB12;

  __BB(12):
L9:
  return _42;

}


