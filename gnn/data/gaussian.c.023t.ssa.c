size_t __GIMPLE (ssa)
ringbuf_size (const size_t n)
{
  size_t size;
  size_t D_7906;
  long unsigned int _1;
  size_t _6;

  __BB(2):
  size_2 = 0ul;
  size_3 = size_2 + 24ul;
  _1 = n_4(D) * 8ul;
  size_5 = size_3 + _1;
  _6 = size_5;
  goto __BB3;

  __BB(3):
L0:
  return _6;

}


size_t __GIMPLE (ssa)
gaussian_size (const size_t n)
{
  size_t size;
  size_t D_8007;
  long unsigned int D_8006;
  long unsigned int _1;
  long unsigned int _8;
  size_t _10;

  __BB(2):
  size_2 = 0ul;
  size_3 = size_2 + 24ul;
  _1 = n_4(D) * 8ul;
  size_5 = size_3 + _1;
  _8 = ringbuf_size (n_4(D));
  size_9 = _8 + size_5;
  _10 = size_9;
  goto __BB3;

  __BB(3):
L0:
  return _10;

}


int __GIMPLE (ssa)
ringbuf_init (const size_t n, struct ringbuf * b)
{
  int D_7908;
  ringbuf_type_t * _1;
  int _2;
  int _10;

  __BB(2):
  _1 = b_3(D) + 24ul;
  b_3(D)->array = _1;
  b_3(D)->head = _Literal (int) -1;
  b_3(D)->tail = 0;
  _2 = (int) n_8(D);
  b_3(D)->size = _2;
  _10 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _10;

}


int __GIMPLE (ssa)
gaussian_init (const size_t n, void * vstate)
{
  struct gaussian_state_t * state;
  int D_8009;
  double * _1;
  double * _2;
  long unsigned int _3;
  struct ringbuf * _4;
  struct ringbuf * _5;
  int _14;

  __BB(2):
  state_7 = vstate_6(D);
  state_7->n = n_9(D);
  _1 = vstate_6(D) + 24ul;
  state_7->window = _1;
  _2 = state_7->window;
  _3 = n_9(D) * 8ul;
  _4 = _2 + _3;
  state_7->rbuf = _4;
  _5 = state_7->rbuf;
  ringbuf_init (n_9(D), _5);
  _14 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _14;

}


int __GIMPLE (ssa)
ringbuf_empty (struct ringbuf * b)
{
  int D_7910;
  int _5;

  __BB(2):
  b_2(D)->head = _Literal (int) -1;
  b_2(D)->tail = 0;
  _5 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _5;

}


int __GIMPLE (ssa)
ringbuf_is_empty (const struct ringbuf * b)
{
  int D_7912;
  int _1;
  _Bool _2;
  int _5;

  __BB(2):
  _1 = b_4(D)->head;
  _2 = _1 == _Literal (const int) -1;
  _5 = (int) _2;
  goto __BB3;

  __BB(3):
L0:
  return _5;

}


int __GIMPLE (ssa)
ringbuf_is_full (const struct ringbuf * b)
{
  int iftmp.0;
  int D_7914;
  int _1;
  int _2;
  int _3;
  int _4;
  int _5;
  int _6;
  int _7;
  int _8;
  int _11;
  int _12;
  int _13;

  __BB(2):
  _1 = b_10(D)->head;
  if (_1 == 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _2 = b_10(D)->tail;
  _3 = b_10(D)->size;
  _4 = _3 + _Literal (int) -1;
  if (_2 == _4)
    goto __BB5;
  else
    goto __BB4;

  __BB(4):
  _5 = b_10(D)->head;
  _6 = b_10(D)->tail;
  _7 = _6 + 1;
  if (_5 == _7)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _12 = 1;
  goto __BB7;

  __BB(6):
  _11 = 0;
  goto __BB7;

  __BB(7):
  _8 = __PHI (__BB5: _12, __BB6: _11);
  _13 = _8;
  goto __BB8;

  __BB(8):
L5:
  return _13;

}


int __GIMPLE (ssa)
ringbuf_insert (const ringbuf_type_t x, struct ringbuf * b)
{
  int D_7937;
  int _1;
  int _2;
  int _3;
  int _4;
  int _5;
  int _6;
  int _7;
  int _8;
  int _9;
  int _10;
  int _11;
  int _12;
  int _13;
  int _14;
  int _15;
  int _16;
  int _17;
  int _18;
  ringbuf_type_t * _19;
  int _20;
  long unsigned int _21;
  long unsigned int _22;
  ringbuf_type_t * _23;
  int _36;

  __BB(2):
  _1 = b_26(D)->head;
  if (_1 == _Literal (int) -1)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  b_26(D)->head = 0;
  b_26(D)->tail = 0;
  goto __BB12;

  __BB(4):
  _2 = b_26(D)->head;
  if (_2 == 0)
    goto __BB5;
  else
    goto __BB8;

  __BB(5):
  _3 = b_26(D)->size;
  _4 = _3 + _Literal (int) -1;
  b_26(D)->head = _4;
  _5 = b_26(D)->tail;
  _6 = b_26(D)->head;
  if (_5 == _6)
    goto __BB6;
  else
    goto __BB12;

  __BB(6):
  _7 = b_26(D)->size;
  if (_7 > 1)
    goto __BB7;
  else
    goto __BB12;

  __BB(7):
  _8 = b_26(D)->tail;
  _9 = _8 + _Literal (int) -1;
  b_26(D)->tail = _9;
  goto __BB12;

  __BB(8):
  _10 = b_26(D)->head;
  _11 = _10 + _Literal (int) -1;
  b_26(D)->head = _11;
  _12 = b_26(D)->tail;
  _13 = b_26(D)->head;
  if (_12 == _13)
    goto __BB9;
  else
    goto __BB12;

  __BB(9):
  _14 = b_26(D)->tail;
  if (_14 == 0)
    goto __BB10;
  else
    goto __BB11;

  __BB(10):
  _15 = b_26(D)->size;
  _16 = _15 + _Literal (int) -1;
  b_26(D)->tail = _16;
  goto __BB12;

  __BB(11):
  _17 = b_26(D)->tail;
  _18 = _17 + _Literal (int) -1;
  b_26(D)->tail = _18;
  goto __BB12;

  __BB(12):
  _19 = b_26(D)->array;
  _20 = b_26(D)->head;
  _21 = (long unsigned int) _20;
  _22 = _21 * 8ul;
  _23 = _19 + _22;
  __MEM <ringbuf_type_t> (_23) = x_34(D);
  _36 = 0;
  goto __BB13;

  __BB(13):
L15:
  return _36;

}


int __GIMPLE (ssa)
gaussian_insert (const gaussian_type_t x, void * vstate)
{
  struct gaussian_state_t * state;
  int D_8011;
  struct ringbuf * _1;
  int _7;

  __BB(2):
  state_3 = vstate_2(D);
  _1 = state_3->rbuf;
  ringbuf_insert (x_5(D), _1);
  _7 = 0;
  goto __BB3;

  __BB(3):
L0:
  return _7;

}


int __GIMPLE (ssa)
ringbuf_pop_back (struct ringbuf * b)
{
  int D_7942;
  int _1;
  int _2;
  int _3;
  int _4;
  int _5;
  int _6;
  int _7;
  int _8;
  int _9;
  int _10;
  int _20;
  int _22;

  __BB(2):
  _1 = ringbuf_is_empty (b_14(D));
  if (_1 != 0)
    goto __BB4;
  else
    goto __BB3;

  __BB(3):
  _2 = b_14(D)->tail;
  if (_2 < 0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  gsl_error ("buffer is empty", ".//ringbuf.c", 132, 19);
  _22 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(5):
  _3 = b_14(D)->head;
  _4 = b_14(D)->tail;
  if (_3 == _4)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  b_14(D)->head = _Literal (int) -1;
  b_14(D)->tail = _Literal (int) -1;
  goto __BB10;

  __BB(7):
  _5 = b_14(D)->tail;
  if (_5 == 0)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  _6 = b_14(D)->size;
  _7 = _6 + _Literal (int) -1;
  b_14(D)->tail = _7;
  goto __BB10;

  __BB(9):
  _8 = b_14(D)->tail;
  _9 = _8 + _Literal (int) -1;
  b_14(D)->tail = _9;
  goto __BB10;

  __BB(10):
  _20 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB11;

  __BB(11):
  _10 = __PHI (__BB4: _22, __BB10: _20);
L9:
  return _10;

}


int __GIMPLE (ssa)
gaussian_delete (void * vstate)
{
  struct gaussian_state_t * state;
  int D_8015;
  struct ringbuf * _1;
  int _2;
  struct ringbuf * _3;
  int _10;

  __BB(2):
  state_6 = vstate_5(D);
  _1 = state_6->rbuf;
  _2 = ringbuf_is_empty (_1);
  if (_2 == 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _3 = state_6->rbuf;
  ringbuf_pop_back (_3);
  goto __BB4;

  __BB(4):
  _10 = 0;
  goto __BB5;

  __BB(5):
L2:
  return _10;

}


ringbuf_type_t __GIMPLE (ssa)
ringbuf_peek (const int i, const struct ringbuf * b)
{
  ringbuf_type_t D_7952;
  int _1;
  ringbuf_type_t * _2;
  int _3;
  int _4;
  int _5;
  int _6;
  long unsigned int _7;
  long unsigned int _8;
  ringbuf_type_t * _9;
  ringbuf_type_t _10;
  ringbuf_type_t _16;
  ringbuf_type_t _18;

  __BB(2):
  _1 = ringbuf_is_empty (b_13(D));
  if (_1 != 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("buffer is empty", ".//ringbuf.c", 159, 19);
  _18 = 1.9e+1;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4):
  _2 = b_13(D)->array;
  _3 = b_13(D)->head;
  _4 = i_15(D) + _3;
  _5 = b_13(D)->size;
  _6 = _4 % _5;
  _7 = (long unsigned int) _6;
  _8 = _7 * 8ul;
  _9 = _2 + _8;
  _16 = __MEM <ringbuf_type_t> ((ringbuf_type_t * const)_9);
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(5):
  _10 = __PHI (__BB3: _18, __BB4: _16);
L2:
  return _10;

}


ringbuf_type_t __GIMPLE (ssa)
ringbuf_peek_front (const struct ringbuf * b)
{
  ringbuf_type_t D_7956;
  int _1;
  ringbuf_type_t * _2;
  int _3;
  long unsigned int _4;
  long unsigned int _5;
  ringbuf_type_t * _6;
  ringbuf_type_t _7;
  ringbuf_type_t _12;
  ringbuf_type_t _14;

  __BB(2):
  _1 = ringbuf_is_empty (b_10(D));
  if (_1 != 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("buffer is empty", ".//ringbuf.c", 172, 19);
  _14 = 1.9e+1;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(4):
  _2 = b_10(D)->array;
  _3 = b_10(D)->head;
  _4 = (long unsigned int) _3;
  _5 = _4 * 8ul;
  _6 = _2 + _5;
  _12 = __MEM <ringbuf_type_t> ((ringbuf_type_t * const)_6);
  // predicted unlikely by early return (on trees) predictor.
  goto __BB5;

  __BB(5):
  _7 = __PHI (__BB3: _14, __BB4: _12);
L2:
  return _7;

}


ringbuf_type_t __GIMPLE (ssa)
ringbuf_peek_back (const struct ringbuf * b)
{
  ringbuf_type_t D_7961;
  int _1;
  int _2;
  ringbuf_type_t * _3;
  int _4;
  long unsigned int _5;
  long unsigned int _6;
  ringbuf_type_t * _7;
  ringbuf_type_t _8;
  ringbuf_type_t _13;
  ringbuf_type_t _15;

  __BB(2):
  _1 = ringbuf_is_empty (b_11(D));
  if (_1 != 0)
    goto __BB4;
  else
    goto __BB3;

  __BB(3):
  _2 = b_11(D)->tail;
  if (_2 < 0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  gsl_error ("buffer is empty", ".//ringbuf.c", 185, 19);
  _15 = 1.9e+1;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB6;

  __BB(5):
  _3 = b_11(D)->array;
  _4 = b_11(D)->tail;
  _5 = (long unsigned int) _4;
  _6 = _5 * 8ul;
  _7 = _3 + _6;
  _13 = __MEM <ringbuf_type_t> ((ringbuf_type_t * const)_7);
  // predicted unlikely by early return (on trees) predictor.
  goto __BB6;

  __BB(6):
  _8 = __PHI (__BB4: _15, __BB5: _13);
L3:
  return _8;

}


size_t __GIMPLE (ssa)
ringbuf_copy (double * dest, const struct ringbuf * b)
{
  int i;
  const int n;
  int iftmp.1;
  size_t D_7966;
  int _1;
  int _2;
  int _3;
  int _4;
  int _5;
  int _6;
  int _7;
  int _8;
  int _9;
  int _10;
  int _11;
  int _12;
  ringbuf_type_t * _13;
  int _14;
  int _15;
  int _16;
  int _17;
  long unsigned int _18;
  long unsigned int _19;
  ringbuf_type_t * _20;
  int _21;
  long unsigned int _22;
  long unsigned int _23;
  sizetype _24;
  double * _25;
  double _26;
  size_t _28;
  int _29;
  int _35;
  int _36;
  size_t _39;
  size_t _43;

  __BB(2):
  _1 = ringbuf_is_empty (b_33(D));
  if (_1 != 0)
    goto __BB4;
  else
    goto __BB3;

  __BB(3):
  _2 = b_33(D)->tail;
  if (_2 < 0)
    goto __BB4;
  else
    goto __BB5;

  __BB(4):
  _43 = 0ul;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB12;

  __BB(5):
  _3 = b_33(D)->head;
  _4 = b_33(D)->tail;
  if (_3 > _4)
    goto __BB6;
  else
    goto __BB7;

  __BB(6):
  _5 = b_33(D)->size;
  _6 = b_33(D)->head;
  _7 = _5 - _6;
  _8 = b_33(D)->tail;
  _9 = _7 + _8;
  _36 = _9 + 1;
  goto __BB8;

  __BB(7):
  _10 = b_33(D)->tail;
  _11 = b_33(D)->head;
  _12 = _10 - _11;
  _35 = _12 + 1;
  goto __BB8;

  __BB(8):
  _29 = __PHI (__BB6: _36, __BB7: _35);
  n_37 = _29;
  i_38 = 0;
  goto __BB10;

  __BB(9):
  _13 = b_33(D)->array;
  _14 = b_33(D)->head;
  _15 = i_27 + _14;
  _16 = b_33(D)->size;
  _17 = _15 % _16;
  _18 = (long unsigned int) _17;
  _19 = _18 * 8ul;
  _20 = _13 + _19;
  _21 = n_37 - i_27;
  _22 = (long unsigned int) _21;
  _23 = _22 * 8ul;
  _24 = _23 + 18446744073709551608ul;
  _25 = dest_40(D) + _24;
  _26 = __MEM <ringbuf_type_t> ((ringbuf_type_t * const)_20);
  __MEM <double> (_25) = _26;
  i_42 = i_27 + 1;
  goto __BB10;

  __BB(10,loop_header(1)):
  i_27 = __PHI (__BB8: i_38, __BB9: i_42);
  if (i_27 < n_37)
    goto __BB9;
  else
    goto __BB11;

  __BB(11):
  _39 = (size_t) n_37;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB12;

  __BB(12):
  _28 = __PHI (__BB4: _43, __BB11: _39);
L9:
  return _28;

}


int __GIMPLE (ssa)
gaussian_get (void * params, gaussian_type_t * result, const void * vstate)
{
  size_t i;
  double sum;
  size_t n;
  const double * kernel;
  const struct gaussian_state_t * state;
  int D_8017;
  struct ringbuf * _1;
  double * _2;
  double * _3;
  long unsigned int _4;
  double * _5;
  double _6;
  long unsigned int _7;
  long unsigned int _8;
  sizetype _9;
  const double * _10;
  double _11;
  double _12;
  int _26;

  __BB(2):
  state_16 = vstate_15(D);
  kernel_18 = params_17(D);
  _1 = state_16->rbuf;
  _2 = state_16->window;
  n_21 = ringbuf_copy (_2, _1);
  sum_22 = 0.0;
  i_23 = 0ul;
  goto __BB4;

  __BB(3):
  _3 = state_16->window;
  _4 = i_14 * 8ul;
  _5 = _3 + _4;
  _6 = __MEM <double> ((double * const)_5);
  _7 = n_21 - i_14;
  _8 = _7 * 8ul;
  _9 = _8 + 18446744073709551608ul;
  _10 = kernel_18 + _9;
  _11 = __MEM <const double> (_10);
  _12 = _6 * _11;
  sum_27 = sum_13 + _12;
  i_28 = i_14 + 1ul;
  goto __BB4;

  __BB(4,loop_header(1)):
  sum_13 = __PHI (__BB2: sum_22, __BB3: sum_27);
  i_14 = __PHI (__BB2: i_23, __BB3: i_28);
  if (i_14 < n_21)
    goto __BB3;
  else
    goto __BB5;

  __BB(5):
  __MEM <gaussian_type_t> (result_24(D)) = sum_13;
  _26 = 0;
  goto __BB6;

  __BB(6):
L3:
  return _26;

}


void __GIMPLE (ssa)
gsl_filter_gaussian_free (struct gsl_filter_gaussian_workspace * w)
{
  double * _1;
  double * _2;
  struct gsl_movstat_workspace * _3;
  struct gsl_movstat_workspace * _4;

  __BB(2):
  _1 = w_8(D)->kernel;
  if (_1 != _Literal (double *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  _2 = w_8(D)->kernel;
  free (_2);
  goto __BB4;

  __BB(4):
  _3 = w_8(D)->movstat_workspace_p;
  if (_3 != _Literal (struct gsl_movstat_workspace *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  _4 = w_8(D)->movstat_workspace_p;
  gsl_movstat_free (_4);
  goto __BB6;

  __BB(6):
  free (w_8(D));
  return;

}


struct gsl_filter_gaussian_workspace * __GIMPLE (ssa)
gsl_filter_gaussian_alloc (const size_t K)
{
  size_t state_size;
  struct gsl_filter_gaussian_workspace * w;
  const size_t H;
  struct gsl_filter_gaussian_workspace * D_7974;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  long unsigned int _4;
  void * _5;
  double * _6;
  long unsigned int _7;
  struct gsl_movstat_workspace * _8;
  struct gsl_movstat_workspace * _9;
  struct gsl_filter_gaussian_workspace * _10;
  struct gsl_filter_gaussian_workspace * _24;
  struct gsl_filter_gaussian_workspace * _27;
  struct gsl_filter_gaussian_workspace * _30;
  struct gsl_filter_gaussian_workspace * _32;

  __BB(2):
  H_13 = K_12(D) / 2ul;
  w_16 = calloc (1ul, 24ul);
  if (w_16 == _Literal (struct gsl_filter_gaussian_workspace *) 0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("failed to allocate space for workspace", ".//gaussian.c", 73, 8);
  _32 = _Literal (struct gsl_filter_gaussian_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(4):
  _1 = H_13 * 2ul;
  _2 = _1 + 1ul;
  w_16->K = _2;
  _3 = w_16->K;
  _4 = _3 * 8ul;
  _5 = malloc (_4);
  w_16->kernel = _5;
  _6 = w_16->kernel;
  if (_6 == _Literal (double *) 0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_filter_gaussian_free (w_16);
  gsl_error ("failed to allocate space for kernel", ".//gaussian.c", 82, 8);
  _30 = _Literal (struct gsl_filter_gaussian_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(6):
  _7 = w_16->K;
  state_size_21 = gaussian_size (_7);
  _8 = gsl_movstat_alloc_with_size (state_size_21, H_13, H_13);
  w_16->movstat_workspace_p = _8;
  _9 = w_16->movstat_workspace_p;
  if (_9 == _Literal (struct gsl_movstat_workspace *) 0)
    goto __BB7;
  else
    goto __BB8;

  __BB(7):
  gsl_filter_gaussian_free (w_16);
  gsl_error ("failed to allocate space for movstat workspace", ".//gaussian.c", 92, 8);
  _27 = _Literal (struct gsl_filter_gaussian_workspace *) 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB9;

  __BB(8):
  _24 = w_16;
  goto __BB9;

  __BB(9):
  _10 = __PHI (__BB3: _32, __BB5: _30, __BB8: _24, __BB7: _27);
L6:
  return _10;

}


int __GIMPLE (ssa)
gsl_filter_gaussian_kernel (const double alpha, const size_t order, const int normalize, struct gsl_vector * kernel)
{
  double * wn;
  double qn;
  double xi;
  double tmp;
  double qm1;
  size_t k;
  double q[11];
  const double beta;
  double gi;
  double yi;
  double xi;
  size_t i;
  double sum;
  const double half;
  const size_t N;
  int D_7993;
  double _1;
  double _2;
  double _3;
  double _4;
  double _5;
  double _6;
  double _7;
  double _8;
  unsigned int _9;
  double _10;
  double _11;
  double _12;
  double _13;
  double _14;
  long unsigned int _15;
  double _16;
  double _17;
  double _18;
  double _19;
  double _20;
  double _21;
  double _22;
  unsigned int _23;
  unsigned int _24;
  int _25;
  double _26;
  double _27;
  int _35;
  int _79;
  int _89;
  int _91;
  int _93;

  __BB(2):
  N_47 = kernel_46(D)->size;
  if (alpha_48(D) <= 0.0)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("alpha must be positive", ".//gaussian.c", 175, 1);
  _93 = 1;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB31;

  __BB(4):
  if (order_49(D) > 10ul)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("derivative order is too large", ".//gaussian.c", 179, 1);
  _91 = 1;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB31;

  __BB(6):
  _1 = (double) N_47;
  _2 = _1 - 1.0e+0;
  half_50 = _2 * 5.0e-1;
  sum_51 = 0.0;
  if (N_47 == 1ul)
    goto __BB7;
  else
    goto __BB11;

  __BB(7):
  if (order_49(D) == 0ul)
    goto __BB8;
  else
    goto __BB9;

  __BB(8):
  gsl_vector_set (kernel_46(D), 0ul, 1.0e+0);
  goto __BB10;

  __BB(9):
  gsl_vector_set (kernel_46(D), 0ul, 0.0);
  goto __BB10;

  __BB(10):
  _89 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB31;

  __BB(11):
  i_52 = 0ul;
  goto __BB13;

  __BB(12):
  _3 = (double) i_29;
  _4 = _3 - half_50;
  xi_80 = _4 / half_50;
  yi_81 = alpha_48(D) * xi_80;
  _5 = yi_81 * -5.0e-1;
  _6 = yi_81 * _5;
  gi_83 = exp (_6);
  gsl_vector_set (kernel_46(D), i_29, gi_83);
  sum_85 = sum_28 + gi_83;
  i_86 = i_29 + 1ul;
  goto __BB13;

  __BB(13,loop_header(1)):
  sum_28 = __PHI (__BB11: sum_51, __BB12: sum_85);
  i_29 = __PHI (__BB11: i_52, __BB12: i_86);
  if (i_29 < N_47)
    goto __BB12;
  else
    goto __BB14;

  __BB(14):
  if (normalize_53(D) != 0)
    goto __BB15;
  else
    goto __BB16;

  __BB(15):
  _7 = 1.0e+0 / sum_28;
  gsl_vector_scale (kernel_46(D), _7);
  goto __BB16;

  __BB(16):
  if (order_49(D) != 0ul)
    goto __BB17;
  else
    goto __BB30;

  __BB(17):
  _8 = alpha_48(D) * -5.0e-1;
  beta_55 = alpha_48(D) * _8;
  _9 = (unsigned int) order_49(D);
  _10 = gsl_pow_uint (half_50, _9);
  _11 = 1.0e+0 / _10;
  q[0] = _11;
  i_58 = 1ul;
  goto __BB19;

  __BB(18):
  q[i_30] = 0.0;
  i_78 = i_30 + 1ul;
  goto __BB19;

  __BB(19,loop_header(2)):
  i_30 = __PHI (__BB17: i_58, __BB18: i_78);
  if (i_30 <= 10ul)
    goto __BB18;
  else
    goto __BB20;

  __BB(20):
  k_59 = 1ul;
  goto __BB25;

  __BB(21):
  qm1_69 = q[0];
  _12 = q[1];
  q[0] = _12;
  i_71 = 1ul;
  goto __BB23;

  __BB(22):
  tmp_73 = q[i_31];
  _13 = (double) i_31;
  _14 = _13 + 1.0e+0;
  _15 = i_31 + 1ul;
  _16 = q[_15];
  _17 = _14 * _16;
  _18 = beta_55 * 2.0e+0;
  _19 = qm1_34 * _18;
  _20 = _17 + _19;
  q[i_31] = _20;
  qm1_75 = tmp_73;
  i_76 = i_31 + 1ul;
  goto __BB23;

  __BB(23,loop_header(5)):
  i_31 = __PHI (__BB21: i_71, __BB22: i_76);
  qm1_34 = __PHI (__BB21: qm1_69, __BB22: qm1_75);
  if (i_31 <= k_33)
    goto __BB22;
  else
    goto __BB24;

  __BB(24):
  k_72 = k_33 + 1ul;
  goto __BB25;

  __BB(25,loop_header(3)):
  k_33 = __PHI (__BB20: k_59, __BB24: k_72);
  if (k_33 <= order_49(D))
    goto __BB21;
  else
    goto __BB26;

  __BB(26):
  i_60 = 0ul;
  goto __BB28;

  __BB(27):
  _21 = (double) i_32;
  _22 = _21 - half_50;
  xi_62 = _22 / half_50;
  _23 = (unsigned int) order_49(D);
  _24 = _23 + 1u;
  _25 = (int) _24;
  qn_64 = gsl_poly_eval (&q, _25, xi_62);
  wn_66 = gsl_vector_ptr (kernel_46(D), i_32);
  _26 = __MEM <double> (wn_66);
  _27 = qn_64 * _26;
  __MEM <double> (wn_66) = _27;
  i_68 = i_32 + 1ul;
  goto __BB28;

  __BB(28,loop_header(4)):
  i_32 = __PHI (__BB26: i_60, __BB27: i_68);
  if (i_32 < N_47)
    goto __BB27;
  else
    goto __BB29;

  __BB(29):
  q ={v} _Literal (double[11]) {CLOBBER};
  goto __BB30;

  __BB(30):
  _79 = 0;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB31;

  __BB(31):
  _35 = __PHI (__BB3: _93, __BB5: _91, __BB10: _89, __BB30: _79);
L28:
  return _35;

}


int __GIMPLE (ssa)
gsl_filter_gaussian (const gsl_filter_end_t endtype, const double alpha, const size_t order, const struct gsl_vector * x, struct gsl_vector * y, struct gsl_filter_gaussian_workspace * w)
{
  struct gsl_vector_view kernel;
  int status;
  int D_7986;
  long unsigned int _1;
  long unsigned int _2;
  long unsigned int _3;
  double * _4;
  struct gsl_movstat_workspace * _5;
  double * _6;
  int _7;
  int _20;
  int _23;
  int _25;

  __BB(2):
  _1 = x_10(D)->size;
  _2 = y_11(D)->size;
  if (_1 != _2)
    goto __BB3;
  else
    goto __BB4;

  __BB(3):
  gsl_error ("input and output vectors must have same length", ".//gaussian.c", 132, 19);
  _25 = 19;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(4):
  if (alpha_12(D) <= 0.0)
    goto __BB5;
  else
    goto __BB6;

  __BB(5):
  gsl_error ("alpha must be positive", ".//gaussian.c", 136, 1);
  _23 = 1;
  // predicted unlikely by early return (on trees) predictor.
  goto __BB7;

  __BB(6):
  _3 = w_13(D)->K;
  _4 = w_13(D)->kernel;
  kernel = gsl_vector_view_array (_4, _3); [return slot optimization]
  gsl_filter_gaussian_kernel (alpha_12(D), order_15(D), 1, &kernel.vector);
  _5 = w_13(D)->movstat_workspace_p;
  _6 = w_13(D)->kernel;
  status_19 = gsl_movstat_apply_accum (endtype_17(D), x_10(D), &gaussian_accum_type, _6, y_11(D), _Literal (struct gsl_vector *) 0, _5);
  _20 = status_19;
  // predicted unlikely by early return (on trees) predictor.
  kernel ={v} _Literal (struct gsl_vector_view) {CLOBBER};
  goto __BB7;

  __BB(7):
  _7 = __PHI (__BB3: _25, __BB5: _23, __BB6: _20);
L5:
  return _7;

}


