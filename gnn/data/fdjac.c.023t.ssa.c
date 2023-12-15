int __GIMPLE (ssa)
fdjac (const struct gsl_vector * x, const struct gsl_vector * wts, struct gsl_multifit_function_fdf * fdf, const struct gsl_vector * f, struct gsl_matrix * J)
{
  double fi;
  double fnext;
  struct gsl_vector_view v;
  double xj;
  double eps;
  const double epsfcn;
  double h;
  size_t j;
  size_t i;
  int status;
  int D_13372;
  int D_13369;
  double iftmp.0;
  double _1;
  double _2;
  double _3;
  double _4;
  long unsigned int _5;
  long unsigned int _6;
  double _11;
  int _12;
  double _18;
  double _19;
  int _25;
  int _36;
  int _50;

  __BB(2):
  status_16 = 0;
  epsfcn_17 = 0.0;
  if (epsfcn_17 > 2.220446049250313080847263336181640625e-16)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _19 = epsfcn_17;
  goto __BB5;

  __BB(4):
  _18 = 2.220446049250313080847263336181640625e-16;
  goto __BB5;

  __BB(5):
  _11 = __PHI (__BB3: _19, __BB4: _18);
  eps_22 = sqrt (_11);
  j_23 = 0ul;
  goto __BB14;

  __BB(6):
  xj_28 = gsl_vector_get (x_26(D), j_9);
  v = gsl_matrix_column (J_29(D), j_9); [return slot optimization]
  _1 = __ABS xj_28;
  h_31 = eps_22 * _1;
  if (h_31 == 0.0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  h_32 = eps_22;
  goto __BB8;

  __BB(8):
  h_10 = __PHI (__BB6: h_31, __BB7: h_32);
  _2 = xj_28 + h_10;
  gsl_vector_set (x_26(D), j_9, _2);
  _36 = gsl_multifit_eval_wf (fdf_24(D), x_26(D), wts_34(D), &v.vector);
  status_37 = _36 + status_7;
  if (status_37 != 0)
    goto __BB9;
  else
    goto __BB10;

  __BB(9):
  _50 = status_37;
  // predicted unlikely by early return (on trees) predictor.
  v ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB16;

  __BB(10):
  gsl_vector_set (x_26(D), j_9, xj_28);
  h_39 = 1.0e+0 / h_10;
  i_40 = 0ul;
  goto __BB12;

  __BB(11):
  fnext_44 = gsl_vector_get (&v.vector, i_8);
  fi_47 = gsl_vector_get (f_45(D), i_8);
  _3 = fnext_44 - fi_47;
  _4 = h_39 * _3;
  gsl_matrix_set (J_29(D), i_8, j_9, _4);
  i_49 = i_8 + 1ul;
  goto __BB12;

  __BB(12,loop_header(2)):
  i_8 = __PHI (__BB10: i_40, __BB11: i_49);
  _5 = fdf_24(D)->n;
  if (i_8 < _5)
    goto __BB11;
  else
    goto __BB13;

  __BB(13):
  v ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  j_42 = j_9 + 1ul;
  goto __BB14;

  __BB(14,loop_header(1)):
  status_7 = __PHI (__BB5: status_16, __BB13: status_37);
  j_9 = __PHI (__BB5: j_23, __BB13: j_42);
  _6 = fdf_24(D)->p;
  if (j_9 < _6)
    goto __BB6;
  else
    goto __BB15;

  __BB(15):
  _25 = status_7;
  goto __BB16;

  __BB(16):
  _12 = __PHI (__BB9: _50, __BB15: _25);
L15:
  return _12;

}


int __GIMPLE (ssa)
gsl_multifit_fdfsolver_dif_df (const struct gsl_vector * x, const struct gsl_vector * wts, struct gsl_multifit_function_fdf * fdf, const struct gsl_vector * f, struct gsl_matrix * J)
{
  int D_13376;
  int _8;

  __BB(2):
  _8 = fdjac (x_2(D), wts_3(D), fdf_4(D), f_5(D), J_6(D));
  goto __BB3;

  __BB(3):
L0:
  return _8;

}


int __GIMPLE (ssa)
gsl_multifit_fdfsolver_dif_fdf (const struct gsl_vector * x, struct gsl_multifit_function_fdf * fdf, struct gsl_vector * f, struct gsl_matrix * J)
{
  int status;
  int D_13380;
  int _1;
  int _13;
  int _14;
  int _15;

  __BB(2):
  status_3 = 0;
  status_9 = gsl_multifit_eval_wf (fdf_5(D), x_6(D), _Literal (const struct gsl_vector *) 0, f_7(D));
  if (status_9 != 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _15 = status_9;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(4):
  status_12 = fdjac (x_6(D), _Literal (const struct gsl_vector *) 0, fdf_5(D), f_7(D), J_10(D));
  if (status_12 != 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _14 = status_12;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(6):
  _13 = status_12;
  goto __BB7;

  __BB(7):
  _1 = __PHI (__BB3: _15, __BB5: _14, __BB6: _13);
L4:
  return _1;

}


