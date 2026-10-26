/** @file hmcode.h Documented includes for HMcode module */

#include "primordial.h"

#ifndef __HMCODE__
#define __HMCODE__

struct fourier_workspace;

struct fourier_growth {
  double * a_table;
  int a_size;
  double * growth_table;
  double * normgrowth_table;
  double * ddnormgrowth_table;
  double * dda_table;

  int gt_size;
  int index_gt_g;
  int index_gt_ddg;
  int index_gt_intg;
  int index_gt_ddintg;
  int index_gt_Omnorad;
  int index_gt_ddOmnorad;

  double Omega_cdm;
  double Omega_b;
  double Omega_v;
  double Omega_w;
  double w0;
  double wa;
  double Omega_m;
  double Omega_nu;
  double a_nu;
  double h;
  double Tcmb;
  double ns;

  double om_m;
  double fnu;

  double a_ini;
  double a_final;

  double smallest_allowed_variation;

  int index_fg_g;
  int index_fg_dg;
  int fg_size;

  int index_norad_H2;
  int index_norad_AH;
  int index_norad_Om;
  int norad_size;
  double* pvecnorad;

  double* P_nw;
  double lnk_norm;

  ErrorMsg error_message; 	/**< zone for writing error messages */

};

/********************************************************************************/

/* @cond INCLUDE_WITH_DOXYGEN */
/*
 * Boilerplate for C++
 */
#ifdef __cplusplus
extern "C" {
#endif

  int fourier_hmcode(
                     struct precision *ppr,
                     struct background *pba,
                     struct perturbations *ppt,
                     struct primordial *ppm,
                     struct fourier *pfo,
                     int index_pk,
                     int index_tau,
                     double tau,
                     double *pk_nl,
                     double **lnpk_l,
                     double **ddlnpk_l,
                     double *k_nl,
                     short * halofit_found_k_max,
                     struct fourier_workspace * pnw
                     );

  int fourier_hmcode_workspace_init(
                                    struct precision *ppr,
                                    struct background *pba,
                                    struct fourier *pfo,
                                    struct fourier_workspace * pnw
                                    );

  int fourier_hmcode_workspace_free(
                                    struct fourier *pfo,
                                    struct fourier_workspace * pnw
                                    );

  int fourier_hmcode_dark_energy_correction(
                                            struct precision *ppr,
                                            struct background *pba,
                                            struct fourier *pfo,
                                            struct fourier_workspace * pnw
                                            );

  int fourier_hmcode_baryonic_feedback(
                                       struct fourier *pfo
                                       );


  int fourier_hmcode_fill_sigtab(
                                 struct precision *ppr,
                                 struct background * pba,
                                 struct perturbations *ppt,
                                 struct primordial * ppm,
                                 struct fourier * pfo,
                                 int index_tau,
                                 int index_pk,
                                 double **lnpk_l,
                                 double **ddlnpk_l,
                                 struct fourier_workspace * pnw
                                 );

  int fourier_hmcode_fill_growtab(
                                  struct precision *ppr,
                                  struct background * pba,
                                  struct fourier * pfo,
                                  struct fourier_workspace * pnw
                                  );

  int fourier_hmcode_growint(
                             struct precision *ppr,
                             struct background * pba,
                             struct fourier * pfo,
                             double a,
                             double w,
                             double wa,
                             double * growth
                             );

  int fourier_hmcode_window_nfw(
                                struct fourier * pfo,
                                double k,
                                double rv,
                                double c,
                                double *window_nfw
                                );

  int fourier_hmcode_halomassfunction(
                                      double nu,
                                      double *hmf
                                      );

  int fourier_hmcode_sigma8_at_z(
                                 struct background *pba,
                                 struct fourier * pfo,
                                 double z,
                                 double * sigma_8,
                                 double * sigma_8_cb,
                                 struct fourier_workspace * pnw
                                 );

  int fourier_hmcode_sigmadisp_at_z(
                                    struct background *pba,
                                    struct fourier * pfo,
                                    double z,
                                    double * sigma_disp,
                                    double * sigma_disp_cb,
                                    struct fourier_workspace * pnw
                                    );

  int fourier_hmcode_sigmadisp100_at_z(
                                       struct background *pba,
                                       struct fourier * pfo,
                                       double z,
                                       double * sigma_disp_100,
                                       double * sigma_disp_100_cb,
                                       struct fourier_workspace * pnw
                                       );

  int fourier_hmcode_sigmaprime_at_z(
                                     struct background *pba,
                                     struct fourier * pfo,
                                     double z,
                                     double * sigma_prime,
                                     double * sigma_prime_cb,
                                     struct fourier_workspace * pnw
                                     );

  int fourier_hmcode_nowiggle_init(
                        struct fourier* pfo,
                        double **lnpk_l,
                        double** ddlnpk_l,
                        int index_pk,
                        struct fourier_workspace * pnw);
  int fourier_hmcode_noradiation_growth_init(struct precision* ppr, struct background* pba,struct fourier* pfo, struct fourier_workspace* pnw);
  int fourier_hmcode_noradiation_growth_free(struct fourier* pfo, struct fourier_workspace* pnw);
  int fourier_hmcode_noradiation_growth_compute( struct fourier* pfo, struct fourier_workspace* pnw);
  int fourier_hmcode_growth_derivs(
                      double a,
                      double* y,
                      double* dy,
                      void * parameters_and_workspace,
                      ErrorMsg error_message
                      );
  int fourier_hmcode_growth_sources(
                       double a,
                       double * y,
                       double * dy,
                       int index_a,
                       void * parameters_and_workspace,
                       ErrorMsg error_message
                       );
  int fourier_hmcode_norad(struct fourier_growth* pfg, double a);
  int hmcode_Pk_nowiggle(struct fourier* pfo, struct fourier_workspace* pnw, double pk_norm);
  double fourier_hmcode_cbnu_ratio(double k, double z, double fnu, double omega_m, double Tcmb, double growth);

#ifdef __cplusplus
}
#endif

#endif
/* @endcond */
