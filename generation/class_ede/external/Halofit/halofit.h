/** @file halofit.h Documented includes for halofit module */

#include "primordial.h"

#ifndef __HALOFIT__
#define __HALOFIT__

enum halofit_integral_type {halofit_integral_one, halofit_integral_two, halofit_integral_three};

/********************************************************************************/

/* @cond INCLUDE_WITH_DOXYGEN */
/*
 * Boilerplate for C++
 */
#ifdef __cplusplus
extern "C" {
#endif
  int fourier_halofit(
                      struct precision *ppr,
                      struct background *pba,
                      struct perturbations *ppt,
                      struct primordial *ppm,
                      struct fourier *pfo,
                      int index_pk,
                      double tau,
                      double *pk_nl,
                      double *lnpk_l,
                      double *ddlnpk_l,
                      double *k_nl,
                      short * halofit_found_k_max
                      );

  int fourier_halofit_integrate(
                                struct fourier *pfo,
                                double * integrand_array,
                                int integrand_size,
                                int ia_size,
                                int index_ia_k,
                                int index_ia_pk,
                                int index_ia_sum,
                                int index_ia_ddsum,
                                double R,
                                enum halofit_integral_type type,
                                double * sum
                                );

#ifdef __cplusplus
}
#endif

#endif
/* @endcond */
