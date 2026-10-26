#include "hmcode.h"
#include "fourier.h"


/**
 * Computes the nonlinear correction on the linear power spectrum, wrapper for 'fourier_hmcode_compute' method.
 *
 * @param ppr Input: pointer to precision structure
 * @param pba Input: pointer to background structure
 * @param ppt Input: pointer to perturbation structure
 * @param ppm Input: pointer to primordial structure
 * @param pfo Input: pointer to fourier structure
 * @param index_pk   Input: index of the pk type, either index_m or index_cb
 * @param index_tau  Input: index of tau, at which to compute the nl correction
 * @param tau        Input: tau, at which to compute the nl correction
 * @param pk_nl      Output:nonlinear power spectrum
 * @param lnpk_l     Input: logarithm of the linear power spectrum for both index_m and index_cb
 * @param ddlnpk_l   Input: spline of the logarithm of the linear power spectrum for both index_m and index_cb
 * @param nl_corr_not_computable_at_this_k Ouput: was the computation doable?
 * @param k_nl       Output: nonlinear scale for index_m and index_cb
 * @param pnw        Input/Output: pointer to nonlinear workspace
 * @return the error status
 */
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
                   short * nl_corr_not_computable_at_this_k,
                   struct fourier_workspace * pnw
                   ) {
  double* pk_nl_baseline;
  double* pk_nl_denominator;
  int index_k;

  if (pfo->hm_version == nl_hmcode_version_2020_baryonic) { //Need to do triple-analysis in this case

    class_call(fourier_hmcode_compute(ppr, pba, ppt, ppm, pfo,
                                      index_pk, index_tau, tau,
                                      pk_nl, lnpk_l, ddlnpk_l,
                                      k_nl, nl_corr_not_computable_at_this_k,
                                      pnw),
               pfo->error_message,
               pfo->error_message);

    class_alloc(pk_nl_baseline,
                pfo->k_size*sizeof(double),
                pfo->error_message);
    pfo->hm_version = nl_hmcode_version_2020;
    class_call(fourier_hmcode_compute(ppr, pba, ppt, ppm, pfo,
                                      index_pk, index_tau, tau,
                                      pk_nl_baseline, lnpk_l, ddlnpk_l,
                                      k_nl, nl_corr_not_computable_at_this_k,
                                      pnw),
               pfo->error_message,
               pfo->error_message);
    pfo->hm_version = nl_hmcode_version_2020_unfitted;
    class_alloc(pk_nl_denominator,
                pfo->k_size*sizeof(double),
                pfo->error_message);
    class_call(fourier_hmcode_compute(ppr, pba, ppt, ppm, pfo,
                                      index_pk, index_tau, tau,
                                      pk_nl_denominator, lnpk_l, ddlnpk_l,
                                      k_nl, nl_corr_not_computable_at_this_k,
                                      pnw),
               pfo->error_message,
               pfo->error_message);
    pfo->hm_version = nl_hmcode_version_2020_baryonic;

    for(index_k=0;index_k<pfo->k_size;++index_k){
      /* printf("k %.10e , Original %.10e , base %.10e , denom %.10e \n",pfo->k[index_k]/0.67,
                                                                      pk_nl[index_k]*pow(pfo->k[index_k],3)/(2.*_PI_*_PI_),
                                                                      pk_nl_baseline[index_k]*pow(pfo->k[index_k],3)/(2.*_PI_*_PI_),
                                                                      pk_nl_denominator[index_k]*pow(pfo->k[index_k],3)/(2.*_PI_*_PI_)); */
      pk_nl[index_k] = pk_nl[index_k] * pk_nl_baseline[index_k] / pk_nl_denominator[index_k];
    }

    free(pk_nl_baseline);
    free(pk_nl_denominator);
  }
  else { //In this case we can do the normal analysis
    class_call(fourier_hmcode_compute(ppr, pba, ppt, ppm, pfo,
                                      index_pk, index_tau, tau,
                                      pk_nl, lnpk_l, ddlnpk_l,
                                      k_nl, nl_corr_not_computable_at_this_k,
                                      pnw),
               pfo->error_message,
               pfo->error_message);
  }
  return _SUCCESS_;
}

/**
 * Computes the nonlinear correction on the linear power spectrum via
 * the method presented in Mead et al. 1505.07833
 *
 * @param ppr Input: pointer to precision structure
 * @param pba Input: pointer to background structure
 * @param ppt Input: pointer to perturbation structure
 * @param ppm Input: pointer to primordial structure
 * @param pfo Input: pointer to fourier structure
 * @param index_pk   Input: index of the pk type, either index_m or index_cb
 * @param index_tau  Input: index of tau, at which to compute the nl correction
 * @param tau        Input: tau, at which to compute the nl correction
 * @param pk_nl      Output:nonlinear power spectrum
 * @param lnpk_l     Input: logarithm of the linear power spectrum for both index_m and index_cb
 * @param ddlnpk_l   Input: spline of the logarithm of the linear power spectrum for both index_m and index_cb
 * @param nl_corr_not_computable_at_this_k Ouput: was the computation doable?
 * @param k_nl       Output: nonlinear scale for index_m and index_cb
 * @param pnw        Input/Output: pointer to nonlinear workspace
 * @return the error status
 */

int fourier_hmcode_compute(
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
                   short * nl_corr_not_computable_at_this_k,
                   struct fourier_workspace * pnw
                   ) {

  /* integers */
  int index_mass, i, ng, nsig;
  int index_k, index_ncol;
  int last_index=0;
  int index_pk_cb;
  int counter, index_nl;

  int index_nu, index_cut;
  int index_y;
  int index_ddy;

  /* Background parameters */
  double Omega_m,fnu,Omega0_m;
  double z_at_tau;
  double rho_crit_today_in_msun_mpc3;
  double growth;
  double anorm;
  double scale_indep_growth;
  double accumulated_growth;
  double Om_norad;

  /* temporary numbers */
  struct fourier_growth* pfg;
  double m, r, nu, sig, sigf;
  double diff, r1, r2, nu1, nu2;
  double k, logk;
  double fac_dewiggle;
  double pk_wig;
  double a,b,h;
  double * head;
  double dv_x,dv_y;
  int abort;
  double tstart;

  /* HMcode parameters */
  double mmin, mmax, nu_min;

  double sigma_disp, sigma_disp100, sigma8, sigma8_cb;
  double delta_c, Delta_v;
  double fraction;

  double sigma_nl, nu_nl, r_nl;
  double sigma_prime;
  double dlnsigdlnR;
  double n_eff;
  double alpha;

  double z_form, g_form;

  double eta;
  double Abary;
  double gst, window_nfw;
  double nu_cut;
  double fac, k_star, fdamp;
  double pk_lin, pk_2h, pk_1h;

  double kdamp;
  double DEcorr;
  double w0, dw_over_da_fld, integral_fld;
  double g_wcdm, g_lcdm;

  /* Baryonic model parameters */
  double sbar_bar, mbar_bar;
  double sbarz_bar, mbarz_bar;
  double sb, mb;
  double ratio_m;
  double fb,fc,fs;
  double DMONLY_halo_mass_fraction;

  /* data fields */
  double * pvecback;
  double * conc;
  double * mass;
  double * sigma_r;
  double * sigmaf_r;
  double * r_virial;
  double * r_real;
  double * nu_arr;

  double * p1h_integrand;

  pfg = pnw->pfg;

  /** include precision parameters that control the number of entries in the growth and sigma tables */
  ng = ppr->n_hmcode_tables;
  nsig = ppr->n_hmcode_tables;

  /** Compute background quantitites today */

  Omega0_m = pba->Omega0_m;
  fnu      = pba->Omega0_ncdm_tot/Omega0_m;

  /** If index_pk_cb, choose Omega0_cb as the matter density parameter.
   * If index_pk_m, choose Omega0_cbn as the matter density parameter. */
  if (index_pk==pfo->index_pk_cluster){
    Omega0_m = Omega0_m - pba->Omega0_ncdm_tot;
  }

  anorm    = 1./(2*pow(_PI_,2));

  /** Call all the relevant background parameters at this tau */
  class_alloc(pvecback,pba->bg_size*sizeof(double),pfo->error_message);

  class_call(background_at_tau(pba,tau,long_info,inter_normal,&last_index,pvecback),
             pba->error_message,
             pfo->error_message);

  Omega_m = pvecback[pba->index_bg_Omega_m];//TBC (i.e. check if for P_cb here we should use Omega_cb) here the total time varying Omega_m is used for delta_c and for Delta_v according to the Mead fit of the Massara simulations.

  /*growth = pvecback[pba->index_bg_D];*/
  class_call(array_spline_hunt(pfg->a_table,
                               pfg->a_size,
                               pvecback[pba->index_bg_a],
                               &last_index,
                               &h,
                               &a,
                               &b,
                               pfo->error_message),
             pfo->error_message,
             pfo->error_message);
  growth = array_spline_eval(pfg->normgrowth_table,pfg->ddnormgrowth_table,last_index,last_index+1,h,a,b); //HM2020

  z_at_tau = 1./pvecback[pba->index_bg_a]-1.;

  /* The number below is the critical density today, rho_c = 3 * H0^2 / 8*pi*G, in units of M_sun over Mpc^3 */
  rho_crit_today_in_msun_mpc3 = 3.*pow(1.e5*pba->h, 2)/8./_PI_/_G_*_Mpc_over_m_/_M_SUN_;

  free(pvecback);


  /* Read additional quantities for HMcode 2020 if required, and setup the 2020 nowiggle workspace */
  if (pfo->hm_version == nl_hmcode_version_2020 || pfo->hm_version == nl_hmcode_version_2020_baryonic || pfo->hm_version == nl_hmcode_version_2020_unfitted){
    class_call(fourier_hmcode_nowiggle_init(pfo,lnpk_l,ddlnpk_l,index_pk,pnw),
               pfo->error_message,
               pfo->error_message);

    head = pfg->growth_table+last_index*pfg->gt_size;
    scale_indep_growth = array_spline_eval(head+pfg->index_gt_g,head+pfg->index_gt_ddg,0,pfg->gt_size,h,a,b);
    accumulated_growth = array_spline_eval(head+pfg->index_gt_intg,head+pfg->index_gt_ddintg,0,pfg->gt_size,h,a,b);
    Om_norad = array_spline_eval(head+pfg->index_gt_Omnorad,head+pfg->index_gt_ddOmnorad,0,pfg->gt_size,h,a,b);
  }
  else{
    scale_indep_growth = 0;
    accumulated_growth = 0;
    Om_norad = 0;
  }

  /** Test whether pk_cb has to be taken into account (only if we have massive neutrinos)*/
  if (pba->has_ncdm==_TRUE_){
    index_pk_cb = pfo->index_pk_cb;
  }
  else {
    index_pk_cb = index_pk;
  }


  /** Get sigma(R=8 Mpc/h), sigma_disp(R=0), sigma_disp(R=100 Mpc/h) and write them into pfo structure */

  class_call(fourier_sigmas(pfo,
                            8./pba->h,
                            lnpk_l[index_pk],ddlnpk_l[index_pk],
                            pfo->k_size_extra,
                            ppr->sigma_k_per_decade,
                            out_sigma,
                            &sigma8),
             pfo->error_message,
             pfo->error_message);

  class_call(fourier_sigmas(pfo,
                            8./pba->h,
                            lnpk_l[index_pk_cb],ddlnpk_l[index_pk_cb],
                            pfo->k_size_extra,
                            ppr->sigma_k_per_decade,
                            out_sigma,
                            &sigma8_cb),
             pfo->error_message,
             pfo->error_message);

  class_call(fourier_sigmas(pfo,
                            0.,
                            lnpk_l[index_pk],ddlnpk_l[index_pk],
                            pfo->k_size_extra,
                            ppr->sigma_k_per_decade,
                            out_sigma_disp,
                            &sigma_disp),
             pfo->error_message,
             pfo->error_message);

  class_call(fourier_sigmas(pfo,
                            100./pba->h,
                            lnpk_l[index_pk],ddlnpk_l[index_pk],
                            pfo->k_size_extra,
                            ppr->sigma_k_per_decade,
                            out_sigma_disp,
                            &sigma_disp100),
             pfo->error_message,
             pfo->error_message);

  pnw->sigma_8[index_pk][index_tau] = sigma8;
  pnw->sigma_disp[index_pk][index_tau] = sigma_disp;
  pnw->sigma_disp_100[index_pk][index_tau] = sigma_disp100;

  /** Initialisation steps for the 1-Halo Power Integral */
  mmin=ppr->mmin_for_p1h_integral/pba->h; //Minimum mass for integration; (unit conversion from  m[Msun/h] to m[Msun]  )
  mmax=ppr->mmax_for_p1h_integral/pba->h; //Maximum mass for integration;

  class_alloc(mass,ppr->nsteps_for_p1h_integral*sizeof(double),pfo->error_message);
  class_alloc(r_real,ppr->nsteps_for_p1h_integral*sizeof(double),pfo->error_message);
  class_alloc(r_virial,ppr->nsteps_for_p1h_integral*sizeof(double),pfo->error_message);
  class_alloc(sigma_r,ppr->nsteps_for_p1h_integral*sizeof(double),pfo->error_message);
  class_alloc(sigmaf_r,ppr->nsteps_for_p1h_integral*sizeof(double),pfo->error_message);
  class_alloc(nu_arr,ppr->nsteps_for_p1h_integral*sizeof(double),pfo->error_message);

  // Linear theory density perturbation threshold for spherical collapse
  // The Einstein de-Sitter value would be delta_c = (3./20.)*pow(12.*_PI_,2./3.) ~1.686
  // Also virialized overdensity
  switch (pfo->hm_version ){
    case nl_hmcode_version_2015:
      delta_c = 1.59+0.0314*log(sigma8); //Mead et al. (2015; arXiv 1505.07833)
      delta_c = delta_c*(1.+0.012299*log10(Omega_m)); //Nakamura & Suto (1997) fitting formula for LCDM models (as in Mead 2016) //BUGFIX :: changed 0.0123 to 0.012299
      delta_c = delta_c*(1.+0.262*fnu); //Mead et al. (2016; arXiv 1602.02154) neutrino addition

      Delta_v=418.*pow(Omega_m, -0.352); //Mead et al. (2015; arXiv 1505.07833)
      Delta_v=Delta_v*(1.+0.916*fnu); //Mead et al. (2016; arXiv 1602.02154) neutrino addition
    break;
    case nl_hmcode_version_2020_baryonic:
    case nl_hmcode_version_2020_unfitted:
    case nl_hmcode_version_2020:
      dv_x = scale_indep_growth; //pfg->index_gt_g is already g(a)/a
      dv_y = accumulated_growth * (1.+z_at_tau);

      delta_c = 1.;
      delta_c = delta_c + (-0.0069 - 0.0208 * (1.-dv_x) + 0.0312 * (1.-dv_x) * (1.-dv_x) + 0.0021 * (1.-dv_y))*pow(log10(Om_norad),1);
      delta_c = delta_c + (+0.0001 - 0.0647 * (1.-dv_x) - 0.0417 * (1.-dv_x) * (1.-dv_x) + 0.0646 * (1.-dv_y))*pow(log10(Om_norad),0);

      delta_c = delta_c*((3./20.)*pow(12.*_PI_,2./3.))*(1.-0.041*fnu);

      Delta_v = 1.;
      Delta_v = Delta_v + (-0.79 - 10.17 * (1.-dv_x) + 2.51 * (1.-dv_x) * (1.-dv_x) +  6.51 * (1.-dv_y))*pow(log10(Om_norad),1);
      Delta_v = Delta_v + (-1.89 +  0.38 * (1.-dv_x) + 18.8 * (1.-dv_x) * (1.-dv_x) - 15.87 * (1.-dv_y))*pow(log10(Om_norad),2);

      Delta_v = Delta_v*(18.*_PI_*_PI_)*(1.+0.763*fnu);
    break;
    //  delta_c = 1.686;
    //  Delta_v = 200.;
    //break;
  }

  // mass or radius fraction respectively
  fraction = pow(0.01, 1./3.);

  /* Fill the arrays needed for the P1H Integral: mass, r_real, r_virial, nu_arr, sigma_r, sigmaf_r
   * The P1H Integral is an integral over nu=delta_c/sigma(M), where M is connected to R via R=(3M)/(4*pi*rho_m).
   * The Integrand is M*Window^2{nu(M)*k, Rv(M), c(M)}*f(nu) with the window being the fouriertransformed
   * NFW profile, Rv = R/Delta_v^(1/3) and Sheth-Thormen halo mass function f.
   * The halo concentration-mass-relation c(M) will be found later.  */

  for (index_mass=0;index_mass<ppr->nsteps_for_p1h_integral;index_mass++){

    m = exp(log(mmin)+log(mmax/mmin)*(index_mass)/(ppr->nsteps_for_p1h_integral-1));
    r = pow((3.*m/(4.*_PI_*rho_crit_today_in_msun_mpc3*Omega0_m)), (1./3.));
    mass[index_mass] = m;
    r_real[index_mass] = r;
    r_virial[index_mass] = r_real[index_mass]/pow(Delta_v, 1./3.);

    class_call(array_interpolate_spline(pnw->rtab,
                                        nsig,
                                        pnw->stab,
                                        pnw->ddstab,
                                        1,
                                        r,
                                        &last_index,
                                        &sig,
                                        1,
                                        pfo->error_message),
               pfo->error_message, pfo->error_message);

    class_call(array_interpolate_spline(pnw->rtab,
                                        nsig,
                                        pnw->stab,
                                        pnw->ddstab,
                                        1,
                                        r*fraction,
                                        &last_index,
                                        &sigf,
                                        1,
                                        pfo->error_message),
               pfo->error_message, pfo->error_message);

    nu=delta_c/sig;
    sigma_r[index_mass] = sig;
    sigmaf_r[index_mass] = sigf;
    nu_arr[index_mass] = nu;
  }

  /** find nonlinear scales k_nl and r_nl and the effective spectral index n_eff */
  nu_nl = 1.;
  nu_min = nu_arr[0];

  /* stop calculating the nonlinear correction if the nonlinear scale is not reached in the table: */
  if (nu_min > nu_nl) {
    if (pfo->fourier_verbose>0) fprintf(stdout, " -> [WARNING:] the minimum mass in the mass-table is too large to find the nonlinear scale at this redshift.\n   Decrease mmin_for_p1h_integral\n");
    * nl_corr_not_computable_at_this_k = _TRUE_;
    free(mass);
    free(r_real);
    free(r_virial);
    free(sigma_r);
    free(sigmaf_r);
    free(nu_arr);
    return _SUCCESS_;
  }

  /* make a first guess for the nonlinear scale */
  class_call(array_interpolate_two_arrays_one_column(
                                                     nu_arr,
                                                     r_real,
                                                     1,
                                                     0,
                                                     ppr->nsteps_for_p1h_integral,
                                                     nu_nl,
                                                     &r_nl,
                                                     pfo->error_message),
             pfo->error_message, pfo->error_message);

  class_call(array_search_bisect(ppr->nsteps_for_p1h_integral,nu_arr,nu_nl,&index_nl,pfo->error_message), pfo->error_message, pfo->error_message);
  /* [NS]:: Possible speedup */
  /*r_nl = (nu_nl-nu_arr[index_nl])/(nu_arr[index_nl+1]-nu_arr[index_nl])*(r_real[index_nl+1]-r_real[index_nl]) + r_real[index_nl];
  class_call(fourier_sigmas(pfo,
                            r_nl,
                            lnpk_l[index_pk_cb],ddlnpk_l[index_pk_cb],
                            pfo->k_size_extra,
                            ppr->sigma_k_per_decade,
                            out_sigma,
                            &sigma_nl),
             pfo->error_message, pfo->error_message);*/

  r1 = r_real[index_nl-1];
  r2 = r_real[index_nl+2];
  nu1 = nu_arr[index_nl-1];
  nu2 = nu_arr[index_nl+2];

  /* As an initial guess, we adopt the linear interpolation */
  r_nl = (nu_nl-nu_arr[index_nl])/(nu_arr[index_nl+1]-nu_arr[index_nl])*(r_real[index_nl+1]-r_real[index_nl]) + r_real[index_nl];
  // for debugging: (if it happens that r_nl is not between r1 and r2, which should never be the case)
  //fprintf(stdout, "r1=%e nu1=%e r2=%e nu2=%e\n", r1, nu_arr[index_nl-1], r2, nu_arr[index_nl+2]);


  // do iteration between r1 and r2 to find the precise value of r_nl --> TODO :: speed up by semi-newtonian iterations
  counter = 0;
  do {

    class_call(fourier_sigmas(pfo,
                              r_nl,
                              lnpk_l[index_pk_cb],ddlnpk_l[index_pk_cb],
                              pfo->k_size_extra,
                              ppr->sigma_k_per_decade,
                              out_sigma,
                              &sigma_nl),
               pfo->error_message, pfo->error_message);

    diff = sigma_nl - delta_c;

    if (diff > ppr->hmcode_tol_sigma){
      r1 = r_nl;
    }
    else if (diff < -ppr->hmcode_tol_sigma) {
      r2 = r_nl;
    }
    r_nl = (r1+r2)/2.;
    counter ++;

    class_test(counter > _MAX_IT_,
               pfo->error_message,
               "could not converge within maximum allowed number of iterations");

  } while (fabs(diff) > ppr->hmcode_tol_sigma);

  if (pfo->fourier_verbose>5){
    fprintf(stdout, "number of iterations for r_nl at z = %e: %d\n", z_at_tau, counter);
  }
  *k_nl = 1./r_nl;

  if (*k_nl > pfo->k[pfo->k_size-1]) {
    * nl_corr_not_computable_at_this_k = _TRUE_;
    free(mass);
    free(r_real);
    free(r_virial);
    free(sigma_r);
    free(sigmaf_r);
    free(nu_arr);
    return _SUCCESS_;
  }
  else {
    * nl_corr_not_computable_at_this_k = _FALSE_;
  }

  /* call sigma_prime function at r_nl to find the effective spectral index n_eff */

  class_call(fourier_sigmas(pfo,
                            r_nl,
                            lnpk_l[index_pk_cb],ddlnpk_l[index_pk_cb],
                            pfo->k_size_extra,
                            ppr->sigma_k_per_decade,
                            out_sigma_prime,
                            &sigma_prime),
             pfo->error_message,
             pfo->error_message);

  dlnsigdlnR = r_nl*pow(sigma_nl, -2)*sigma_prime;
  n_eff = -3.- dlnsigdlnR;

  pnw->sigma_prime[index_pk][index_tau] = sigma_prime;

  /** Calculate halo concentration-mass relation conc(mass) (Bullock et al. 2001) */
  class_alloc(conc,ppr->nsteps_for_p1h_integral*sizeof(double),pfo->error_message);

  switch (pfo->hm_version ){
    case nl_hmcode_version_2015:
      DEcorr = pnw->dark_energy_correction;
    break;
    case nl_hmcode_version_2020:
    //  DEcorr = pnw->dark_energy_correction;
    //break;
    case nl_hmcode_version_2020_baryonic:
    case nl_hmcode_version_2020_unfitted:
      /* Dark Energy Dolag correction */
      class_call(fourier_hmcode_growint(ppr,pba,pfo,1./(1.+z_at_tau),-1.,0.,&g_lcdm),
                 pfo->error_message, pfo->error_message);
      /*class_call(background_w_fld(pba,1.,&w0,&dw_over_da_fld,&integral_fld),
                 pba->error_message,
                 pfo->error_message);

      class_call(fourier_hmcode_growint(ppr,pba,pfo,1./(1.+z_at_tau),w0,dw_over_da_fld*(-1.),&g_wcdm),
                 pfo->error_message,
                 pfo->error_message);
      DEcorr = pnw->dark_energy_correction*g_lcdm/g_wcdm;*/
      ////printf("DEcorr = %.10e (%.10e %.10e) -- w %.10e l %.10e, sig = %.5e, a %.5e\n",DEcorr,pnw->dark_energy_correction,g_lcdm/g_wcdm,g_wcdm,g_lcdm,scale_indep_growth/(1.+z_at_tau)/7.11270e-01,1./(z_at_tau+1.)); //TODO :: replace with normgrowth
      //DEcorr = pnw->dark_energy_correction*g_lcdm/(scale_indep_growth/(1.+z_at_tau)/7.11270e-01);
      DEcorr = pnw->dark_energy_correction*g_lcdm/growth;
      //printf("DEcorr = %.10e ,  a %.5e\n",DEcorr,1./(z_at_tau+1.));
    break;
  }
  switch (pfo->hm_version ){
    case nl_hmcode_version_2015:
      Abary = pfo->c_min;
      alpha = 3.24 * pow(1.85, n_eff);
    break;
    case nl_hmcode_version_2020:
      Abary = ( 5.1958429 * pow(sigma8_cb,0.) + 0. ) * pow(10,z_at_tau*0.);
      alpha = 1.8751465*pow(1.6029913,n_eff);
    break;
    case nl_hmcode_version_2020_baryonic:
      Abary = ( (-0.496*(pfo->log10T_heat_hmcode-7.8)+3.44) * pow(sigma8_cb,0.) + 0.) * pow(10,z_at_tau*(-0.0371*(pfo->log10T_heat_hmcode-7.8)-0.0671));
      alpha = 1.0;
    break;
    case nl_hmcode_version_2020_unfitted:
      Abary = 4.;
      alpha = 1.;
    break;
  }

  /* Fill concentration array */
  for (index_mass=0;index_mass<ppr->nsteps_for_p1h_integral;index_mass++){
    //find growth rate at formation
    g_form = delta_c*growth/sigmaf_r[index_mass];
    if (g_form > 1.) g_form = 1.;

    /*class_call(array_interpolate_two_arrays_one_column(
                                                       pnw->growtable,
                                                       pnw->ztable,
                                                       1,
                                                       0,
                                                       ng,
                                                       g_form,
                                                       &z_form,
                                                       pfo->error_message),
               pfo->error_message, pfo->error_message);*/

    class_call(array_spline_hunt(pfg->normgrowth_table,
                                 pfg->a_size,
                                 g_form,
                                 &last_index,
                                 &h,
                                 &a,
                                 &b,
                                 pfo->error_message),
               pfo->error_message,
               pfo->error_message);
    double a_form = array_spline_eval(pfg->a_table,pfg->dda_table,last_index,last_index+1,h,a,b);
    z_form = 1./a_form-1.; //HM2020

    if (z_form < z_at_tau){
      conc[index_mass] = Abary;
    } else {
      conc[index_mass] = Abary*(1.+z_form)/(1.+z_at_tau);
    }
    if(z_at_tau < pfo->z_infinity){
      conc[index_mass] = conc[index_mass]*DEcorr;
    }
  }

  /* HMcode parameters for non-linear corerction */
  switch (pfo->hm_version ){
    case nl_hmcode_version_2015:
      k_star=0.584/sigma_disp;   // Damping wavenumber of the 1-halo term at very large scales;
      eta = pfo->eta_0 - 0.3*sigma8; // halo bloating parameter
      fdamp = 0.0095*pow(sigma_disp100*pba->h, 1.37);
    break;
    case nl_hmcode_version_2020:
    //In CAMB, these are cb sigma8
      eta = 0.1281210*pow(sigma8_cb, -0.3643963);
      k_star=0.0561778*pow(sigma8_cb,-1.0131066)*pba->h;
      fdamp = 0.2695822 * pow(sigma8_cb, 0.9403087);
      kdamp = 0.0569871*pow(sigma8_cb, -1.0890162) *pba->h;
    break;
    case nl_hmcode_version_2020_baryonic:
    case nl_hmcode_version_2020_unfitted:
      eta = 0.;
      k_star = 0.;
      kdamp = 0.;
      fdamp = 0.;
    break;
  }

  // Damping factor for 2-halo term
   if (fdamp<1.e-3) fdamp=1.e-3;
   if (fdamp>0.99)  fdamp=0.99;
  if ( pfo->hm_version == nl_hmcode_version_2020_baryonic || pfo->hm_version == nl_hmcode_version_2020_unfitted ){
    fdamp = 0.;
  }

  /* the 1h integral contains the halo mass function proportional to exp(-nu^2).
   * To save time, the integration loop cuts, when nu exceeds a large value,
   * where the integrand is 0 anyhow. This cut index is found here. */
  nu_cut = 10.;
  //if (nu_cut < nu_arr[ppr->nsteps_for_p1h_integral-1]){
  if (nu_cut < nu_arr[ppr->nsteps_for_p1h_integral-1] && _FALSE_){
    class_call(array_search_bisect(ppr->nsteps_for_p1h_integral,nu_arr,nu_cut,&index_cut,pfo->error_message), pfo->error_message, pfo->error_message);
  }
  else {
    index_cut = ppr->nsteps_for_p1h_integral;
  }

  i=0;
  index_nu=i;
  i++;
  index_y=i;
  i++;
  index_ddy=i;
  i++;
  index_ncol=i;

#pragma omp parallel firstprivate(pba, pfo, pnw,index_nu,index_y,index_ddy,index_ncol,lnpk_l,index_pk,anorm,conc,r_virial,eta,nu_arr,fnu,mass,k_star,rho_crit_today_in_msun_mpc3,Omega0_m,fdamp,sigma_disp,last_index,kdamp,alpha,pk_nl,index_cut,z_at_tau,m) private(k,logk,pk_lin,index_mass,window_nfw,gst,p1h_integrand,pk_1h,fac,pk_2h,pk_wig,fac_dewiggle,tstart,sbar_bar,sbarz_bar,mbar_bar,mbarz_bar,mb,sb,ratio_m,fb,fc,fs,DMONLY_halo_mass_fraction) shared(abort) default(none)
  {

#ifdef _OPENMP
  tstart = omp_get_wtime();
#endif

#pragma omp for
  for (index_k = 0; index_k < pfo->k_size; index_k++){

    k = pfo->k[index_k];
    logk = log(k);

    class_alloc_parallel(p1h_integrand,index_cut*index_ncol*sizeof(double),pfo->error_message);

    pk_lin = exp(lnpk_l[index_pk][index_k])*pow(k,3)*anorm; //convert P_k to Delta_k^2

    for (index_mass=0; index_mass<index_cut; index_mass++){ //Calculates the integrand for the ph1 integral at all nu values
      //get the nu^eta-value of the window
      class_call_parallel(fourier_hmcode_window_nfw(
                                           pfo,
                                           pow(nu_arr[index_mass], eta)*k,
                                           r_virial[index_mass],
                                           conc[index_mass], //(rv/rs)
                                           &window_nfw),
                 pfo->error_message, pfo->error_message);
      //get the value of the halo mass function
      class_call_parallel(fourier_hmcode_halomassfunction( //equivalent to g_nu function (or g_ST)
                                                 nu_arr[index_mass],
                                                 &gst),
                 pfo->error_message, pfo->error_message);

      p1h_integrand[index_mass*index_ncol+index_nu] = nu_arr[index_mass];

      switch (pfo->hm_version ){
        case nl_hmcode_version_2015:
        break;
        case nl_hmcode_version_2020_unfitted:
        case nl_hmcode_version_2020:
          window_nfw *= (1.-fnu);
        break;
        case nl_hmcode_version_2020_baryonic:
          sbar_bar = -0.0030*(pfo->log10T_heat_hmcode-7.8)+0.0201;
          sbarz_bar = 0.0224*(pfo->log10T_heat_hmcode-7.8)+0.409;
          sb = MIN(sbar_bar*pow(10,z_at_tau*sbarz_bar),pba->Omega0_b/pba->Omega0_m);

          mbar_bar = pow(10.,1.81*(pfo->log10T_heat_hmcode-7.8)+13.87);
          mbarz_bar = 0.195*(pfo->log10T_heat_hmcode-7.8)-0.108;

          mb = mbar_bar*pow(10,mbarz_bar*z_at_tau);
          ratio_m = pow(pba->h*mass[index_mass]/mb,2.); //hmod%nbar
          fb = pba->Omega0_b/pba->Omega0_m;
          fc = pba->Omega0_cdm/pba->Omega0_m;
          fs = sb;
          DMONLY_halo_mass_fraction = fc+(fb-fs)*ratio_m/(1.+ratio_m);

          window_nfw = window_nfw*DMONLY_halo_mass_fraction; //Account for gas expulsion

          window_nfw = window_nfw+sb; //Account for star formation
          break;
      }

      p1h_integrand[index_mass*index_ncol+index_y] = mass[index_mass]*gst*pow(window_nfw, 2.);
      //if ((tau==pba->conformal_age) && (index_k == 0)) {
      //fprintf(stdout, "%d %e %e\n", index_cut, p1h_integrand[index_mass*index_ncol+index_nu], p1h_integrand[index_mass*index_ncol+index_y]);
      //}
    }

    class_call_parallel(array_spline(p1h_integrand,
                            index_ncol,
                            index_cut,
                            index_nu,
                            index_y,
                            index_ddy,
                            _SPLINE_EST_DERIV_,
                            pfo->error_message),
               pfo->error_message,
               pfo->error_message);

    class_call_parallel(array_integrate_all_trapzd_or_spline(
                                                    p1h_integrand,
                                                    index_ncol,
                                                    index_cut,
                                                    index_cut-1, //ranges from 0 to n-1
                                                    index_nu,
                                                    index_y,
                                                    index_ddy,
                                                    &pk_1h,
                                                    pfo->error_message),
               pfo->error_message,
               pfo->error_message);

    switch (pfo->hm_version ){
      case nl_hmcode_version_2015:
        if (pow(k/k_star, 2)>7.){
          fac = 1.;     //prevents problems if (k/k*)^2 is large
        }
        else{
          fac = 1.-exp(-pow((k/k_star), 2.));
        }
      break;
      case nl_hmcode_version_2020:
        if ( k_star == 0 ){
          fac = 1.;
        } else {
          fac = pow(k/k_star,4)/(1.+pow(k/k_star,4));
        }
      break;
      case nl_hmcode_version_2020_unfitted:
      case nl_hmcode_version_2020_baryonic:
        fac = 1.;
      break;
    }

    pk_1h = pk_1h*anorm*pow(k,3)*fac/(rho_crit_today_in_msun_mpc3*Omega0_m);  // dimensionless power

    switch (pfo->hm_version ){
      case nl_hmcode_version_2015:
        pk_2h = pk_lin;
        if(fdamp>0){
          pk_2h = pk_2h*(1.-fdamp*pow(tanh(k*sigma_disp/sqrt(fdamp)), 2.));
        }
      break;
      case nl_hmcode_version_2020:
      case nl_hmcode_version_2020_unfitted:
      case nl_hmcode_version_2020_baryonic:
        if(logk<pnw->lnk_wiggle[0] || logk>pnw->lnk_wiggle[pfo->nk_wiggle-1]){
          pk_wig = 0.;
        }
        else{
          class_call_parallel(array_interpolate_spline(pnw->lnk_wiggle,
		                                                   pfo->nk_wiggle,
		                                                   pnw->pk_wiggle,
		                                                   pnw->ddpk_wiggle,
		                                                   1,
		                                                   log(k),
		                                                   &last_index,
		                                                   &pk_wig,
		                                                   1,
		                                                   pfo->error_message),
			               pfo->error_message,
			               pfo->error_message);
			  }
        fac_dewiggle = exp(-(k*k*sigma_disp*sigma_disp));
        pk_2h = pk_lin + (fac_dewiggle-1.)*pk_wig;


        if(fdamp>0){
          pk_2h = pk_2h*(1.-fdamp*(pow(k/kdamp,2.8534197))/(1.+pow(k/kdamp,2.8534197)));
        }
      break;
    }

    //p_hm function: Combine pk_1h and pk_2h
    class_test_parallel(pk_2h < 0. || pk_1h < 0.,pfo->error_message,"The 2 halo or 1 halo term is negative for HMcode, and the 'safe_negative' option is disabled. Aborting");
    pk_nl[index_k] = pow((pow(pk_1h, alpha) + pow(pk_2h, alpha)), (1./alpha))/pow(k,3)/anorm;
    free(p1h_integrand);

#pragma omp flush(abort)

  }

#ifdef _OPENMP
        if (pfo->fourier_verbose>2)
          printf("In %s: time spent in parallel region (loop over k's) = %e s for thread %d\n",
                 __func__,omp_get_wtime()-tstart,omp_get_thread_num());
#endif
  } /* end of parallel region */
  if (abort == _TRUE_) return _FAILURE_;

  // print parameter values
  if ((pfo->fourier_verbose > 2 && tau==pba->conformal_age) || pfo->fourier_verbose > 3){
    fprintf(stdout, " -> Parameters at redshift z = %e:\n", z_at_tau);
    fprintf(stdout, "    fnu:		%e\n", fnu);
    fprintf(stdout, "    sigd [Mpc/h]:	%e\n", sigma_disp*pba->h);
    //fprintf(stdout, "    sigd100 [Mpc/h]:    %e\n", sigma_disp100*pba->h);
    fprintf(stdout, "    sigma8:		%e\n", sigma8);
    fprintf(stdout, "    nu min:		%e\n", nu_arr[0]);
    fprintf(stdout, "    nu max:		%e\n", nu_arr[ppr->nsteps_for_p1h_integral-1]);
    fprintf(stdout, "    r_v min [Mpc/h]:    %e\n", r_virial[0]*pba->h);
    fprintf(stdout, "    r_v max [Mpc/h]:    %e\n", r_virial[ppr->nsteps_for_p1h_integral-1]*pba->h);
    fprintf(stdout, "    r_nl [Mpc/h]:	%e\n", r_nl*pba->h);
    fprintf(stdout, "    k_nl [h/Mpc]:	%e\n", *k_nl/pba->h);
    fprintf(stdout, "    sigma_nl:		%e\n", sigma_nl/delta_c);
    fprintf(stdout, "    neff:		%e\n", n_eff);
    fprintf(stdout, "    c min:		%e\n", conc[ppr->nsteps_for_p1h_integral-1]);
    fprintf(stdout, "    c max:		%e\n", conc[0]);
    fprintf(stdout, "    Dv:			%e\n", Delta_v);
    fprintf(stdout, "    dc:			%e\n", delta_c);
    fprintf(stdout, "    eta:		%e\n", eta);
    fprintf(stdout, "    k*:			%e\n", k_star/pba->h);
    fprintf(stdout, "    Abary:		%e\n", Abary);
    fprintf(stdout, "    fdamp:		%e\n", fdamp);
    fprintf(stdout, "    alpha:		%e\n", alpha);
    fprintf(stdout, "    ksize, kmin, kmax:   %d, %e, %e\n", pfo->k_size, pfo->k[0]/pba->h, pfo->k[pfo->k_size-1]/pba->h);

  }

  free(conc);
  free(mass);
  free(r_real);
  free(r_virial);
  free(sigma_r);
  free(sigmaf_r);
  free(nu_arr);

  return _SUCCESS_;
}

/**
 * allocate and fill arrays of nonlinear workspace (currently used only by HMcode)
 *
 * @param ppr         Input: pointer to precision structure
 * @param pba         Input: pointer to background structure
 * @param pfo         Input: pointer to fourier structure
 * @param pnw         Output: pointer to nonlinear workspace
 * @return the error status
 */

int fourier_hmcode_workspace_init(
                                  struct precision *ppr,
                                  struct background *pba,
                                  struct fourier *pfo,
                                  struct fourier_workspace * pnw
                                  ){

  int ng;
  int index_pk;

  /** - allocate arrays of the nonlinear workspace */

  class_alloc(pnw->rtab,ppr->n_hmcode_tables*sizeof(double),pfo->error_message);
  class_alloc(pnw->stab,ppr->n_hmcode_tables*sizeof(double),pfo->error_message);
  class_alloc(pnw->ddstab,ppr->n_hmcode_tables*sizeof(double),pfo->error_message);

  ng = ppr->n_hmcode_tables;

  class_alloc(pnw->growtable,ng*sizeof(double),pfo->error_message);
  class_alloc(pnw->ztable,ng*sizeof(double),pfo->error_message);
  class_alloc(pnw->tautable,ng*sizeof(double),pfo->error_message);

  class_alloc(pnw->sigma_8,pfo->pk_size*sizeof(double *),pfo->error_message);
  class_alloc(pnw->sigma_disp,pfo->pk_size*sizeof(double *),pfo->error_message);
  class_alloc(pnw->sigma_disp_100,pfo->pk_size*sizeof(double *),pfo->error_message);
  class_alloc(pnw->sigma_prime,pfo->pk_size*sizeof(double *),pfo->error_message);

  for (index_pk=0; index_pk<pfo->pk_size; index_pk++){
    class_alloc(pnw->sigma_8[index_pk],pfo->tau_size*sizeof(double),pfo->error_message);
    class_alloc(pnw->sigma_disp[index_pk],pfo->tau_size*sizeof(double),pfo->error_message);
    class_alloc(pnw->sigma_disp_100[index_pk],pfo->tau_size*sizeof(double),pfo->error_message);
    class_alloc(pnw->sigma_prime[index_pk],pfo->tau_size*sizeof(double),pfo->error_message);
  }

  /** - fill table with scale independent growth factor */

  class_call(fourier_hmcode_fill_growtab(ppr,pba,pfo,pnw),
             pfo->error_message,
             pfo->error_message);

  //pfo->nk_wiggle = 512;//512; //TODO :: do correctly
  class_alloc(pnw->pk_wiggle, pfo->nk_wiggle*sizeof(double), pfo->error_message);
  class_alloc(pnw->ddpk_wiggle, pfo->nk_wiggle*sizeof(double), pfo->error_message);
  class_alloc(pnw->lnk_wiggle, pfo->nk_wiggle*sizeof(double), pfo->error_message);

  return _SUCCESS_;
}

/**
 * deallocate arrays in the nonlinear worksapce (currently used only
 * by HMcode)
 *
 * @param pfo Input: pointer to fourier structure
 * @param pnw Input: pointer to nonlinear workspace
 * @return the error status
 */

int fourier_hmcode_workspace_free(
                                  struct fourier *pfo,
                                  struct fourier_workspace * pnw
                                  ) {
  int index_pk;

  free(pnw->rtab);
  free(pnw->stab);
  free(pnw->ddstab);

  free(pnw->growtable);
  free(pnw->ztable);
  free(pnw->tautable);

  for (index_pk=0; index_pk<pfo->pk_size; index_pk++){
    free(pnw->sigma_8[index_pk]);
    free(pnw->sigma_disp[index_pk]);
    free(pnw->sigma_disp_100[index_pk]);
    free(pnw->sigma_prime[index_pk]);
  }

  free(pnw->sigma_8);
  free(pnw->sigma_disp);
  free(pnw->sigma_disp_100);
  free(pnw->sigma_prime);

  free(pnw->pk_wiggle);
  free(pnw->ddpk_wiggle);
  free(pnw->lnk_wiggle);

  return _SUCCESS_;
}

/**
 * set the HMcode dark energy correction (if w is not -1)
 *
 * @param ppr         Input: pointer to precision structure
 * @param pba         Input: pointer to background structure
 * @param pfo         Input: pointer to fourier structure
 * @param pnw         Output: pointer to nonlinear workspace
 * @return the error status
 */

int fourier_hmcode_dark_energy_correction(
                                          struct precision *ppr,
                                          struct background *pba,
                                          struct fourier *pfo,
                                          struct fourier_workspace * pnw
                                          ) {

  int last_index;
  double * pvecback;
  double tau_growth;
  double g_lcdm,g_wcdm;
  double w0,dw_over_da_fld,integral_fld;
  double h,a,b;

  /** - if there is dynamical Dark Energy (w is not -1) modeled as a fluid */

  if (pba->has_fld==_TRUE_){

    class_alloc(pvecback,pba->bg_size*sizeof(double),pfo->error_message);

    class_call(background_tau_of_z(
                                   pba,
                                   pfo->z_infinity,
                                   &tau_growth
                                   ),
               pba->error_message,
               pfo->error_message);

    class_call(background_at_tau(pba,tau_growth,long_info,inter_normal,&last_index,pvecback),
               pba->error_message,
               pfo->error_message);

    class_call(background_w_fld(pba,1.,&w0,&dw_over_da_fld,&integral_fld),
               pba->error_message,
               pfo->error_message);

    class_call(fourier_hmcode_growint(ppr,pba,pfo,1./(1.+pfo->z_infinity),-1.,0.,&g_lcdm),
               pfo->error_message, pfo->error_message);

    class_call(fourier_hmcode_growint(ppr,pba,pfo,1./(1.+pfo->z_infinity),w0,dw_over_da_fld*(-1.),&g_wcdm),
               pfo->error_message,
               pfo->error_message);

    free(pvecback);

    switch( pfo->hm_version ){
      case nl_hmcode_version_2015:
        pnw->dark_energy_correction = pow(g_wcdm/g_lcdm, 1.5);
      break;
      case nl_hmcode_version_2020:
      case nl_hmcode_version_2020_unfitted:
      case nl_hmcode_version_2020_baryonic:
        //pnw->dark_energy_correction = g_wcdm/g_lcdm; //Here, we are missing the redshift dependence, which we include only within the respective function
        //fourier_hmcode_norad(pnw->pfg,1./(pfo->z_infinity+1.));
        class_call(array_spline_hunt(pnw->pfg->a_table,
                                     pnw->pfg->a_size,
                                     1./(pfo->z_infinity+1.),
                                     &last_index,
                                     &h,
                                     &a,
                                     &b,
                                     pfo->error_message),
                   pfo->error_message,
                   pfo->error_message);

        double growth = array_spline_eval(pnw->pfg->normgrowth_table,pnw->pfg->ddnormgrowth_table,last_index,last_index+1,h,a,b);
        //double* head = pnw->pfg->growth_table+last_index*pnw->pfg->gt_size;
        //double scale_indep_growth = array_spline_eval(head+pnw->pfg->index_gt_g,head+pnw->pfg->index_gt_ddg,0,pnw->pfg->gt_size,h,a,b);//TODO :: replace with normgrowth
        ////printf("DE CORR = %.10e %.10e -> %.10e (grow = %.10e)\n",g_wcdm,g_lcdm,g_wcdm/g_lcdm,scale_indep_growth/(1.+pfo->z_infinity)/7.11270e-01);
        pnw->dark_energy_correction = growth/g_lcdm;//(scale_indep_growth/(1.+pfo->z_infinity)/7.11270e-01)/g_lcdm; //HM2020
      break;
    }
  }

  /** - otherwise, we assume no dynamical Dark Energy (w is -1) */

  else {
    pnw->dark_energy_correction = 1.;
  }

  return _SUCCESS_;
}

/**
 * set the HMcode baryonic feedback parameters according to the chosen feedback model
 *
 * @param pfo   Output: pointer to fourier structure
 * @return the error status
 */
int fourier_hmcode_baryonic_feedback(
                                     struct fourier *pfo
                                     ) {

  switch (pfo->feedback) {

  case nl_emu_dmonly:
    {
      pfo->eta_0 = 0.603;
      pfo->c_min = 3.13;
      break;
    }

  case nl_owls_dmonly:
    {
      pfo->eta_0 = 0.64;
      pfo->c_min = 3.43;
      break;
    }

  case nl_owls_ref:
    {
      pfo->eta_0 = 0.68;
      pfo->c_min = 3.91;
      break;
    }

  case nl_owls_agn:
    {
      pfo->eta_0 = 0.76;
      pfo->c_min = 2.32;
      break;
    }

  case nl_owls_dblim:
    {
      pfo->eta_0 = 0.70;
      pfo->c_min = 3.01;
      break;
    }

  case nl_user_defined:
    {
      /* eta_0 and c_min already passed in input */
      break;
    }
  }
  return _SUCCESS_;
}

/**
 * Function that fills pnw->rtab, pnw->stab and pnw->ddstab with (r,
 * sigma, ddsigma) logarithmically spaced in r.  Called by
 * fourier_init at for all tau to account for scale-dependant growth
 * before fourier_hmcode is called
 *
 * @param ppr Input: pointer to precision structure
 * @param pba Input: pointer to background structure
 * @param ppt Input: pointer to perturbation structure
 * @param ppm Input: pointer to primordial structure
 * @param pfo Input: pointer to fourier structure
 * @param index_tau  Input: index of tau, at which to compute the nl correction
 * @param lnpk_l   Input: logarithm of the linear power spectrum for either index_m or index_cb
 * @param ddlnpk_l Input: spline of the logarithm of the linear power spectrum for either index_m or index_cb
 * @param pnw Output: pointer to nonlinear workspace
 * @return the error status
 */

int fourier_hmcode_fill_sigtab(
                               struct precision * ppr,
                               struct background * pba,
                               struct perturbations * ppt,
                               struct primordial * ppm,
                               struct fourier * pfo,
                               int index_tau,
                               int index_pk,
                               double **lnpk_l,
                               double **ddlnpk_l,
                               struct fourier_workspace * pnw
                               ) {

  double r;
  double rmin, rmax;
  double sig;
  double * sigtab;
  double * lpk, *ddlpk;
  int i, index_r, index_sig, index_ddsig, index_n, nsig;
  double cbcorr, z, growth;
  int last_index = 0;
  double a,b,h;
  double* head;

  struct fourier_growth * pfg = pnw->pfg;

  rmin = ppr->rmin_for_sigtab/pba->h;
  rmax = ppr->rmax_for_sigtab/pba->h;
  nsig = ppr->n_hmcode_tables;

  i=0;
  index_r=i;
  i++;
  index_sig=i;
  i++;
  index_ddsig=i;
  i++;
  index_n=i;

  class_alloc((sigtab),(nsig*index_n*sizeof(double)),pfo->error_message);
  class_alloc(lpk, pfo->k_size_extra*sizeof(double), pfo->error_message);
  class_alloc(ddlpk, pfo->k_size_extra*sizeof(double), pfo->error_message);
  for (int index_k=0; index_k<pfo->k_size_extra; index_k++){
    cbcorr = 0.;
    if(index_pk == 1 && (pfo->hm_version == nl_hmcode_version_2020 || pfo->hm_version == nl_hmcode_version_2020_baryonic)){
      class_call(background_z_of_tau(pba,pfo->tau[index_tau],&z),
                 pba->error_message,
                 pfg->error_message);
      class_call(array_spline_hunt(pfg->a_table,
                                   pfg->a_size,
                                   1./(z+1.),
                                   &last_index,
                                   &h,
                                   &a,
                                   &b,
                                   pfo->error_message),
                 pfo->error_message,
                 pfo->error_message);

      head = pfg->growth_table+last_index*pfg->gt_size;
      growth = array_spline_eval(head+pfg->index_gt_g,head+pfg->index_gt_ddg,0,pfg->gt_size,h,a,b);
      // growth = g(a)/a = 1 in the EdS case, otherwise it is found in the growth table (without radiation, though)
      cbcorr = 2. * log(fourier_hmcode_cbnu_ratio(pfo->k[index_k],z,pfg->fnu,pfg->om_m,pfg->Tcmb,growth/(1.+z)));
      lpk[index_k] = lnpk_l[index_pk][index_k]+cbcorr;
    }
    else{
      lpk[index_k] = lnpk_l[0][index_k];//Always take the P(k)_cb if we are a) in the cb case or b) not in HMcode 2020
    }
  }

  class_call(array_spline_table_columns(
		             pfo->ln_k,
		             pfo->k_size_extra,
		             lpk,
		             1,
		             ddlpk,
		             _SPLINE_EST_DERIV_,
		             pfo->error_message),
		         pfo->error_message,
		         pfo->error_message);

  for (i=0;i<nsig;i++){
    r=exp(log(rmin)+log(rmax/rmin)*i/(nsig-1));

    class_call(fourier_sigmas(pfo,
                              r,
                              lpk,
                              ddlpk,
                              pfo->k_size_extra,
                              ppr->sigma_k_per_decade,
                              out_sigma,
                              &sig),
               pfo->error_message,
               pfo->error_message);

    sigtab[i*index_n+index_r]=r;
    sigtab[i*index_n+index_sig]=sig;
  }

  class_call(array_spline(sigtab,
						  index_n,
						  nsig,
						  index_r,
						  index_sig,
						  index_ddsig,
						  _SPLINE_EST_DERIV_,
						  pfo->error_message),
             pfo->error_message,
             pfo->error_message);
  if (index_tau == pfo->tau_size-1){
    for (i=0;i<nsig;i++){
      pnw->rtab[i] = sigtab[i*index_n+index_r];
      pnw->stab[i] = sigtab[i*index_n+index_sig];
      pnw->ddstab[i] = sigtab[i*index_n+index_ddsig];
    }
  }
  else{
    for (i=0;i<nsig;i++){
      pnw->stab[i] = sigtab[i*index_n+index_sig];
      pnw->ddstab[i] = sigtab[i*index_n+index_ddsig];
    }
  }

  free(lpk);
  free(ddlpk);
  free(sigtab);

  return _SUCCESS_;
}


/**
 * Function that fills pnw->tautable and pnw->growtable with (tau, D(tau))
 * linearly spaced in scalefactor a.
 * Called by fourier_init at before the loop over tau
 *
 * @param ppr Input: pointer to precision structure
 * @param pba Input: pointer to background structure (will provide the scale independent growth factor)
 * @param pfo Input/Output: pointer to fourier structure
 * @param pnw Output: pointer to nonlinear workspace
 * @return the error status
 */

int fourier_hmcode_fill_growtab(
                                struct precision * ppr,
                                struct background * pba,
                                struct fourier * pfo,
                                struct fourier_workspace * pnw
                                ){

  double z, ainit, amax, scalefactor, tau_growth;
  int index_scalefactor, last_index, ng;
  double * pvecback;

  ng = ppr->n_hmcode_tables;
  ainit = ppr->ainit_for_growtab;
  amax = ppr->amax_for_growtab;

  last_index = 0;

  class_alloc(pvecback,pba->bg_size*sizeof(double),pfo->error_message);

  for (index_scalefactor=0;index_scalefactor<ng;index_scalefactor++){
    scalefactor = ainit+(amax-ainit)*(index_scalefactor)/(ng-1);
    z = 1./scalefactor-1.;

    pnw->ztable[index_scalefactor] = z;

    class_call(background_tau_of_z(
                                   pba,
                                   z,
                                   &tau_growth
                                   ),
               pba->error_message, pfo->error_message);

    pnw->tautable[index_scalefactor] = tau_growth;

    class_call(background_at_tau(pba,tau_growth,long_info,inter_normal,&last_index,pvecback),
               pba->error_message,
               pfo->error_message);

    pnw->growtable[index_scalefactor] = pvecback[pba->index_bg_D];

  }

  free(pvecback);

  return _SUCCESS_;
}

/**
 * This function finds the scale independent growth factor by
 * integrating the approximate relation d(lnD)/d(lna) =
 * Omega_m(z)^gamma by Linder & Cahn 2007
 *
 * @param ppr Input: pointer to precision structure
 * @param pba Input: pointer to background structure
 * @param pfo Input: pointer to fourier structure
 * @param a   Input: scalefactor
 * @param w0  Input: dark energy equation of state today
 * @param wa  Input: dark energy equation of state varying with a: w=w0+(1-a)wa
 * @param growth Output: scale independent growth factor at a
 * @return the error status
 */

int fourier_hmcode_growint(
                           struct precision * ppr,
                           struct background * pba,
                           struct fourier * pfo,
                           double a,
                           double w0,
                           double wa,
                           double * growth
                           ){

  double z, ainit, amax, scalefactor, gamma, X_de, Hubble2, Omega_m;
  int i, index_scalefactor, index_a, index_growth, index_ddgrowth, index_gcol, ng; // index_scalefactor is a running index while index_a is a column index
  double * pvecback;
  double * integrand;

  ng = 1024; // number of growth values (stepsize of the integral), should not be hardcoded and replaced by a precision parameter
  ainit = a;
  amax = 1.;

  i=0;
  index_a = i;
  i++;
  index_growth = i;
  i++;
  index_ddgrowth = i;
  i++;
  index_gcol = i;

  class_alloc(integrand,ng*index_gcol*sizeof(double),pfo->error_message);
  class_alloc(pvecback,pba->bg_size*sizeof(double),pfo->error_message);

  if (ainit == amax) {
    *growth = 1.;
  }
  else {

    for (index_scalefactor=0;index_scalefactor<ng;index_scalefactor++){

      scalefactor = ainit+(amax-ainit)*(index_scalefactor)/(ng-1);
      z = 1./scalefactor-1.;

      /* This will compute Omega_m(z) for the input values of w0 and wa, to let the user compare the wCDM and LCDM cases. This is why we cannot extract Omega_m(z) fromn the background module in this place. */
      X_de = pow(scalefactor, -3.*(1.+w0+wa))*exp(-3.*wa*(1.-scalefactor));
      Hubble2 = (pba->Omega0_m*pow((1.+z), 3.) + pba->Omega0_k*pow((1.+z), 2.) + pba->Omega0_de*X_de);
      Omega_m = (pba->Omega0_m*pow((1.+z), 3.))/Hubble2;
      /* Samuel brieden: TBC: check that the matching between the
         background quantity and this fitting formula improves by
         using Omega_cb (as it is done in background). Carefull:
         Hubble remains with Omega0_m */

      if (w0 == -1.){
        gamma = 0.55;
      }
      else if (w0 < -1.){
        gamma = 0.55+0.02*(1+w0);
      }
      else {
        gamma = 0.55+0.05*(1+w0);
      }
      integrand[index_scalefactor*index_gcol+index_a] = scalefactor;
      integrand[index_scalefactor*index_gcol+index_growth]= -pow(Omega_m, gamma)/scalefactor;
    }

    class_call(array_spline(integrand,
                            index_gcol,
                            ng,
                            index_a,
                            index_growth,
                            index_ddgrowth,
                            _SPLINE_EST_DERIV_,
                            pfo->error_message),
               pfo->error_message,
               pfo->error_message);

    class_call(array_integrate_all_trapzd_or_spline(integrand,
                                                    index_gcol,
                                                    ng,
                                                    0, //ng-1,
                                                    index_a,
                                                    index_growth,
                                                    index_ddgrowth,
                                                    growth,
                                                    pfo->error_message),
               pfo->error_message,
               pfo->error_message);

    *growth = exp(*growth);

  }
  //fprintf(stdout, "%e %e \n", a, *growth);
  free(pvecback);
  free(integrand);

  return _SUCCESS_;
}

/**
 * This is the fourier transform of the NFW density profile.
 *
 * @param pfo Input: pointer to fourier structure
 * @param k   Input: wave vector
 * @param rv  Input: virial radius
 * @param c   Input: concentration = rv/rs (with scale radius rs)
 * @param window_nfw Output: Window Function of the NFW profile
 * @return the error status
 */

int fourier_hmcode_window_nfw(
                              struct fourier * pfo,
                              double k,
                              double rv,
                              double c,
                              double *window_nfw
                              ){
  double si1, si2, ci1, ci2, ks;
  double p1, p2, p3;

  ks = k*rv/c;

  class_call(sine_integral(
                           ks*(1.+c),
                           &si2,
                           pfo->error_message
                           ),
             pfo->error_message, pfo->error_message);

  class_call(sine_integral(
                           ks,
                           &si1,
                           pfo->error_message
                           ),
             pfo->error_message, pfo->error_message);

  class_call(cosine_integral(
                             ks*(1.+c),
                             &ci2,
                             pfo->error_message
                             ),
             pfo->error_message, pfo->error_message);

  class_call(cosine_integral(
                             ks,
                             &ci1,
                             pfo->error_message
                             ),
             pfo->error_message, pfo->error_message);

  p1=cos(ks)*(ci2-ci1);
  p2=sin(ks)*(si2-si1);
  p3=sin(ks*c)/(ks*(1.+c));

  *window_nfw=p1+p2-p3;
  *window_nfw=*window_nfw/(log(1.+c)-c/(1.+c));

  return _SUCCESS_;
}

/**
 * This is the Sheth-Tormen halo mass function (1999, MNRAS, 308, 119)
 *
 * @param nu   Input: the \f$ \nu \f$ parameter that depends on the halo mass via \f$ \nu(M) = \delta_c/\sigma(M) \f$
 * @param hmf  Output: Value of the halo mass function at this \f$ \nu \f$
 * @return the error status
 */

int fourier_hmcode_halomassfunction(
                                    double nu,
                                    double * hmf
                                    ){

  double p, q, A;

  p=0.3;
  q=0.707;
  A=0.21616;

  *hmf=A*(1.+(pow(q*nu*nu, -p)))*exp(-q*nu*nu/2.);

  return _SUCCESS_;
}

/**
 * Compute sigma8(z)
 *
 * @param pba        Input: pointer to background structure
 * @param pfo        Input: pointer to fourier structure
 * @param z          Input: redshift
 * @param sigma_8    Output: sigma8(z)
 * @param sigma_8_cb Output: sigma8_cb(z)
 * @param pnw        Output: pointer to nonlinear workspace
 * @return the error status
 */

int fourier_hmcode_sigma8_at_z(
                               struct background *pba,
                               struct fourier * pfo,
                               double z,
                               double * sigma_8,
                               double * sigma_8_cb,
                               struct fourier_workspace * pnw
                               ) {

  double tau;

  class_call(background_tau_of_z(pba,
                                 z,
                                 &tau),
             pba->error_message,
             pfo->error_message);

  if (pfo->tau_size == 1) {
    *sigma_8 = pnw->sigma_8[pfo->index_pk_m][0];
  }
  else {
    class_call(array_interpolate_two(pfo->tau,
                                     1,
                                     0,
                                     pnw->sigma_8[pfo->index_pk_m],
                                     1,
                                     pfo->tau_size,
                                     tau,
                                     sigma_8,
                                     1,
                                     pfo->error_message),
               pfo->error_message,
               pfo->error_message);
  }


  if (pba->has_ncdm == _TRUE_){

    if (pfo->tau_size == 1) {
      *sigma_8_cb = pnw->sigma_8[pfo->index_pk_cb][0];
    }
    else {
      class_call(array_interpolate_two(pfo->tau,
                                       1,
                                       0,
                                       pnw->sigma_8[pfo->index_pk_cb],
                                       1,
                                       pfo->tau_size,
                                       tau,
                                       sigma_8_cb,
                                       1,
                                       pfo->error_message),
                 pfo->error_message,
                 pfo->error_message);
    }

  }
  else{
    *sigma_8_cb = *sigma_8;
  }



  return _SUCCESS_;
}

/**
 * Compute sigmadisp(z)
 *
 * @param pba           Input: pointer to background structure
 * @param pfo           Input: pointer to fourier structure
 * @param z             Input: redshift
 * @param sigma_disp    Output: sigmadisp(z)
 * @param sigma_disp_cb Output: sigmadisp_cb(z)
 * @param pnw           Output: pointer to nonlinear workspace
 * @return the error status
 */

int fourier_hmcode_sigmadisp_at_z(
                                  struct background *pba,
                                  struct fourier * pfo,
                                  double z,
                                  double * sigma_disp,
                                  double * sigma_disp_cb,
                                  struct fourier_workspace * pnw
                                  ) {

  double tau;

  class_call(background_tau_of_z(pba,
                                 z,
                                 &tau),
             pba->error_message,
             pfo->error_message);

  if (pfo->tau_size == 1) {
    *sigma_disp = pnw->sigma_disp[pfo->index_pk_m][0];
  }
  else {
    class_call(array_interpolate_two(pfo->tau,
                                     1,
                                     0,
                                     pnw->sigma_disp[pfo->index_pk_m],
                                     1,
                                     pfo->tau_size,
                                     tau,
                                     sigma_disp,
                                     1,
                                     pfo->error_message),
               pfo->error_message,
               pfo->error_message);
  }

  if (pba->has_ncdm == _TRUE_){

    if (pfo->tau_size == 1) {
      *sigma_disp_cb = pnw->sigma_disp[pfo->index_pk_cb][0];
    }
    else {
      class_call(array_interpolate_two(pfo->tau,
                                       1,
                                       0,
                                       pnw->sigma_disp[pfo->index_pk_cb],
                                       1,
                                       pfo->tau_size,
                                       tau,
                                       sigma_disp_cb,
                                       1,
                                       pfo->error_message),
                 pfo->error_message,
                 pfo->error_message);
    }

  }
  else{
    *sigma_disp_cb = *sigma_disp;
  }



  return _SUCCESS_;
}

/**
 * Compute sigmadisp100(z)
 *
 * @param pba               Input: pointer to background structure
 * @param pfo               Input: pointer to fourier structure
 * @param z                 Input: redshift
 * @param sigma_disp_100    Output: sigmadisp100(z)
 * @param sigma_disp_100_cb Output: sigmadisp100_cb(z)
 * @param pnw           Output: pointer to nonlinear workspace
 * @return the error status
 */

int fourier_hmcode_sigmadisp100_at_z(
                                     struct background *pba,
                                     struct fourier * pfo,
                                     double z,
                                     double * sigma_disp_100,
                                     double * sigma_disp_100_cb,
                                     struct fourier_workspace * pnw
                                     ) {

  double tau;

  class_call(background_tau_of_z(pba,
                                 z,
                                 &tau),
             pba->error_message,
             pfo->error_message);

  if (pfo->tau_size == 1) {
    *sigma_disp_100 = pnw->sigma_disp_100[pfo->index_pk_m][0];
  }
  else {
    class_call(array_interpolate_two(pfo->tau,
                                     1,
                                     0,
                                     pnw->sigma_disp_100[pfo->index_pk_m],
                                     1,
                                     pfo->tau_size,
                                     tau,
                                     sigma_disp_100,
                                     1,
                                     pfo->error_message),
               pfo->error_message,
               pfo->error_message);
  }

  if (pba->has_ncdm == _TRUE_){

    if (pfo->tau_size == 1) {
      *sigma_disp_100_cb = pnw->sigma_disp_100[pfo->index_pk_cb][0];
    }
    else {
      class_call(array_interpolate_two(pfo->tau,
                                       1,
                                       0,
                                       pnw->sigma_disp_100[pfo->index_pk_cb],
                                       1,
                                       pfo->tau_size,
                                       tau,
                                       sigma_disp_100_cb,
                                       1,
                                       pfo->error_message),
                 pfo->error_message,
                 pfo->error_message);
    }

  }
  else{
    *sigma_disp_100_cb = *sigma_disp_100;
  }


  return _SUCCESS_;
}

/**
 * Compute sigma'(z)
 *
 * @param pba            Input: pointer to background structure
 * @param pfo            Input: pointer to fourier structure
 * @param z              Input: redshift
 * @param sigma_prime    Output: sigma'(z)
 * @param sigma_prime_cb Output: sigma'_cb(z)
 * @param pnw            Output: pointer to nonlinear workspace
 * @return the error status
 */

int fourier_hmcode_sigmaprime_at_z(
                                   struct background *pba,
                                   struct fourier * pfo,
                                   double z,
                                   double * sigma_prime,
                                   double * sigma_prime_cb,
                                   struct fourier_workspace * pnw
                                   ) {

  double tau;

  class_call(background_tau_of_z(pba,
                                 z,
                                 &tau),
             pba->error_message,
             pfo->error_message);

  if (pfo->tau_size == 1) {
    *sigma_prime = pnw->sigma_prime[pfo->index_pk_m][0];
  }
  else {
    class_call(array_interpolate_two(pfo->tau,
                                     1,
                                     0,
                                     pnw->sigma_prime[pfo->index_pk_m],
                                     1,
                                     pfo->tau_size,
                                     tau,
                                     sigma_prime,
                                     1,
                                     pfo->error_message),
               pfo->error_message,
               pfo->error_message);
  }

  if (pba->has_ncdm == _TRUE_){

    if (pfo->tau_size == 1) {
      *sigma_prime_cb = pnw->sigma_prime[pfo->index_pk_cb][0];
    }
    else {
      class_call(array_interpolate_two(pfo->tau,
                                       1,
                                       0,
                                       pnw->sigma_prime[pfo->index_pk_cb],
                                       1,
                                       pfo->tau_size,
                                       tau,
                                       sigma_prime_cb,
                                       1,
                                       pfo->error_message),
                 pfo->error_message,
                 pfo->error_message);
    }

  }
  else{
    *sigma_prime_cb = *sigma_prime;
  }


  return _SUCCESS_;
}


int fourier_hmcode_nowiggle_init(
                                 struct fourier* pfo,
                                 double **lnpk_l,
                                 double** ddlnpk_l,
                                 int index_pk,
                                 struct fourier_workspace * pnw
                                 ) {
  int i;
  double lnpk;
  //double wiggle_dx = 0.2;
  double wiggle_sigma = 0.25;
  int wiggle_smooth = 0;//dewiggle_Gaussian
  int divide_by_nowiggle = _TRUE_; //TODO :: should be true
  double knorm_dewiggle = 0.03;
  double kmin_wiggle = 5e-3; //Mpc/h
  double logkmin_wiggle = log(kmin_wiggle);
  double kmax_wiggle = 5.;//5.;
  double logkmax_wiggle = log(kmax_wiggle);
  int last_index = 0;
  double loganorm = log(1/(2.*_PI_*_PI_));
  double* pk_lin_wiggled;
  // These values are fundamentally important for the wiggling!

  double* lnpk_lin;
  pnw->pfg->lnk_norm = log(0.03/pnw->pfg->h); //Mpc/h

  class_alloc(lnpk_lin, pfo->nk_wiggle*sizeof(double), pfo->error_message);
  class_alloc(pk_lin_wiggled, pfo->nk_wiggle*sizeof(double), pfo->error_message);
  class_alloc(pnw->pfg->P_nw, pfo->nk_wiggle*sizeof(double), pfo->error_message);
  for(i=0;i<pfo->nk_wiggle;++i){
    pnw->lnk_wiggle[i] = logkmin_wiggle+i*(logkmax_wiggle-logkmin_wiggle)/(pfo->nk_wiggle-1);
    class_call(array_interpolate_spline(pfo->ln_k,
                                        pfo->k_size_extra,
                                        lnpk_l[index_pk],
                                        ddlnpk_l[index_pk],
                                        1,
                                        pnw->lnk_wiggle[i],
                                        &last_index,
                                        &lnpk,
                                        1,
                                        pfo->error_message),
               pfo->error_message,
               pfo->error_message);
    lnpk_lin[i] = lnpk+3.*pnw->lnk_wiggle[i]+loganorm;
  }
  if (divide_by_nowiggle == _TRUE_) {
    double pk_norm;
    class_call(array_interpolate_spline(pfo->ln_k,
                                        pfo->k_size_extra,
                                        lnpk_l[index_pk],
                                        ddlnpk_l[index_pk],
                                        1,
                                        pnw->pfg->lnk_norm,
                                        &last_index,
                                        &pk_norm,
                                        1,
                                        pfo->error_message),
               pfo->error_message,
               pfo->error_message);
     class_call(hmcode_Pk_nowiggle(pfo, pnw,pk_norm),
                pfo->error_message,
                pfo->error_message);

     for(i=0;i<pfo->nk_wiggle;++i){
        pk_lin_wiggled[i] = exp(lnpk_lin[i])/pnw->pfg->P_nw[i];
     }
     //class_stop(pfo->error_message,"Divide by nowiggle not implemented yet");
  }
  else{
    for(i=0;i<pfo->nk_wiggle;++i){
        pk_lin_wiggled[i] = lnpk_lin[i];
     }
  }

  if (wiggle_smooth == 0) {//dewiggle_Gaussian
    class_call(array_smooth_Gaussian(pnw->lnk_wiggle,
                                     pk_lin_wiggled,
                                     pnw->pk_wiggle, //is being overwritten here!
                                     pfo->nk_wiggle,
                                     wiggle_sigma,
                                     pfo->error_message),
               pfo->error_message,
               pfo->error_message);
  }
  //TODO :: other implementations
  // ELSE IF (wiggle_smooth == dewiggle_tophat) THEN
  //    CALL smooth_array_tophat(log(k), Pk_smt(:, ia), wiggle_dx)

  if (divide_by_nowiggle == _TRUE_) {

     for(i=0;i<pfo->nk_wiggle;++i){
        pnw->pk_wiggle[i] = pnw->pk_wiggle[i]*pnw->pfg->P_nw[i];
     }
     //pnw->pk_wiggle = pnw->pk_wiggle*Pk_nwg;
     //class_stop(pfo->error_message,"Divide by nowiggle not implemented yet");
  }
  else{
    for(i=0;i<pfo->nk_wiggle;++i){
      pnw->pk_wiggle[i] = exp(pnw->pk_wiggle[i]);
    }
  }

  for(i=0;i<pfo->nk_wiggle;++i){
    pnw->pk_wiggle[i] = exp(lnpk_lin[i])-pnw->pk_wiggle[i];
  }
  class_call(array_spline_table_columns(pnw->lnk_wiggle,
                                        pfo->nk_wiggle,
                                        pnw->pk_wiggle,
                                        1,
                                        pnw->ddpk_wiggle,
                                        _SPLINE_EST_DERIV_,
                                        pfo->error_message),
             pfo->error_message,
             pfo->error_message);
  free(lnpk_lin);
  free(pk_lin_wiggled);
  free(pnw->pfg->P_nw);
  return _SUCCESS_;
}

int hmcode_Pk_nowiggle(struct fourier* pfo, struct fourier_workspace* pnw, double pk_norm){

  struct fourier_growth* pfg = pnw->pfg;
  int index_k;
  double wm,wb,rb;
  double s, alpha, Gamma, q,L,C,Tk_nw,Pk_nw;
  double pk_nw_norm;
  double k_norm_nowiggle = 0.03;
  double * ddP_nw;
  double k;
  int last_index = 0;

  class_alloc(ddP_nw, pfo->nk_wiggle*sizeof(double), pfo->error_message);

  wm = pfg->Omega_m*pfg->h*pfg->h;
  wb = pfg->Omega_b*pfg->h*pfg->h;
  rb = wb/wm;

  s = 44.5*log(9.83/wm)/sqrt(1.+10.*pow(wb,0.75));
  alpha = 1.-0.328*log(431.*wm)*rb+0.38*log(22.3*wm)*rb*rb;

  for(index_k=0;index_k<pfo->nk_wiggle;++index_k){
    k = exp(pnw->lnk_wiggle[index_k]);
    Gamma = (wm/pfg->h)*(alpha+(1.-alpha)/(1.+pow(0.43*k*s,4)));
    q = k/pfg->h*(pfg->Tcmb/2.7)*(pfg->Tcmb/2.7)/Gamma;
    L = log(2.*exp(1.)+1.8*q);
    C = 14.2+731./(1.+62.5*q);
    Tk_nw = L/(L+C*q*q);
    pfg->P_nw[index_k] = pow(k,pfg->ns+3.)*pow(Tk_nw,2);//pow(k/kpiv,pfg->ns)*pow(Tk_nw,2);
    //printf("k = %.10e , P_nw = %.10e \n",k,pnw->pfg->P_nw[index_k]);
  }

  class_call(array_spline_table_columns(pnw->lnk_wiggle,
                                        pfo->nk_wiggle,
                                        pfg->P_nw,
                                        1,
                                        ddP_nw,
                                        _SPLINE_EST_DERIV_,
                                        pfo->error_message),
             pfo->error_message,
             pfo->error_message);
  class_call(array_interpolate_spline(pnw->lnk_wiggle,
                                      pfo->nk_wiggle,
                                      pfg->P_nw,
                                      ddP_nw,
                                      1,
                                      pfg->lnk_norm,
                                      &last_index,
                                      &pk_nw_norm,
                                      1,
                                      pfo->error_message),
             pfo->error_message,
             pfo->error_message);
  for(index_k=0;index_k<pfo->nk_wiggle;++index_k){
    pfg->P_nw[index_k] = pfg->P_nw[index_k]*pk_norm/pk_nw_norm;
    //printf("k = %.10e , P_nw = %.10e \n",exp(pnw->lnk_wiggle[index_k]),pnw->pfg->P_nw[index_k]);
  }
  free(ddP_nw);
  return _SUCCESS_;
}

int fourier_hmcode_noradiation_growth_init(struct precision* ppr, struct background* pba,struct fourier* pfo, struct fourier_workspace* pnw){
  struct fourier_growth* pfg;
  int index_fg,index_gt, index_norad;
  int i;

  class_alloc(pfg,sizeof(struct fourier_growth),pfo->error_message);
  pnw->pfg = pfg;

  index_fg = 0;
  class_define_index(pfg->index_fg_g,_TRUE_,index_fg,1);
  class_define_index(pfg->index_fg_dg,_TRUE_,index_fg,1);

  pfg->fg_size = index_fg;

  index_gt = 0;
  class_define_index(pfg->index_gt_g,_TRUE_,index_gt,1); //Actually g(a)/a
  class_define_index(pfg->index_gt_ddg,_TRUE_,index_gt,1);
  class_define_index(pfg->index_gt_intg,_TRUE_,index_gt,1);
  class_define_index(pfg->index_gt_ddintg,_TRUE_,index_gt,1);
  class_define_index(pfg->index_gt_Omnorad,_TRUE_,index_gt,1);
  class_define_index(pfg->index_gt_ddOmnorad,_TRUE_,index_gt,1);

  pfg->gt_size = index_gt;

  index_norad = 0;
  class_define_index(pfg->index_norad_H2,_TRUE_,index_norad,1);
  class_define_index(pfg->index_norad_AH,_TRUE_,index_norad,1);
  class_define_index(pfg->index_norad_Om,_TRUE_,index_norad,1);

  pfg->norad_size = index_norad;


  double a_ini = 1e-4;
  double a_final = 1;
  double lna_ini = log(a_ini);
  double lna_final = log(a_final);

  pfg->a_size = 128;
  class_alloc(pfg->a_table,pfg->a_size*sizeof(double),pfo->error_message);
  class_alloc(pfg->growth_table,pfg->a_size*pfg->gt_size*sizeof(double),pfo->error_message);
  class_alloc(pfg->normgrowth_table,pfg->a_size*sizeof(double),pfo->error_message);
  class_alloc(pfg->ddnormgrowth_table,pfg->a_size*sizeof(double),pfo->error_message);
  class_alloc(pfg->dda_table,pfg->a_size*sizeof(double),pfo->error_message);
  class_alloc(pfg->pvecnorad,pfg->norad_size*sizeof(double),pfo->error_message);
  for(i=0;i<pfg->a_size;++i){
    pfg->a_table[i] = a_ini*exp(i*(lna_final-lna_ini)/(pfg->a_size-1));
  }
  pfg->a_ini = a_ini;
  pfg->a_final = a_final;
  pfg->a_table[0]=a_ini;
  pfg->a_table[pfg->a_size-1]=a_final;
  pfg->smallest_allowed_variation = ppr->smallest_allowed_variation;

  /* Apparently the neutrino mass is set purposefully to 0 in the original HMcode to derive a LCDM-equivalent cosmology. Here we follow the same procedure */
  double mnu = 0.;
  double neutrino_constant;
  neutrino_constant = (14./11.)*180.*1.202056903159594285399738161511449990764986292/(7.*pow(_PI_,4));
  neutrino_constant = neutrino_constant*5.670367e-8*pow(pba->T_cmb,3)/(1.38064852e-23*(3.*pow(100./(1e3*149597870700*60.*60.*180./_PI_),2)/(8.*_PI_*6.67408e-11))*2.99792458e8);
  neutrino_constant = neutrino_constant*1.60218e-19/2.99792458e8/2.99792458e8;
  neutrino_constant = 1./neutrino_constant;

  pfg->Omega_cdm = pba->Omega0_m-pba->Omega0_b;//pba->Omega0_cdm;
  pfg->Omega_b = pba->Omega0_b;
  pfg->Omega_v = pba->Omega0_scf;
  double Omega_nu_rad = pba->Omega0_g*pba->Neff*7./8.*pow(4./11.,4./3.);
  if(mnu==0){ //TODO :: can this be removed, since mnu=0 always?
    pfg->Omega_nu = Omega_nu_rad;
    pfg->a_nu = 1;
  }
  else{
    pfg->a_nu = pba->T_cmb*pow(4./11.,1./3.) /(mnu/pba->Neff)*(1.38064852e-23/1.60218e-19);
    pfg->Omega_nu = Omega_nu_rad*(pow(1.+pow(0.3173*1./pfg->a_nu,1.83),1./1.83));
  }
  pfg->Omega_w = pba->Omega0_fld;
  pfg->w0 = pba->w0_fld;
  pfg->wa = pba->wa_fld;
  pfg->Omega_m = 1.-pfg->Omega_v-pfg->Omega_w;
  pfg->h = pba->h;
  pfg->Tcmb = pba->T_cmb;
  pfg->ns = 0.95; //TODO :: assign consistently
  pfg->om_m = pba->Omega0_m*pba->h*pba->h;
  pfg->fnu = pba->Omega0_ncdm_tot/pba->Omega0_m;
  // class_test(pba->Omega0_scf>0,
  //            pfo->error_message,
  //            "Cannot use scalar field scf with HMcode (not yet coded).");

  class_call(fourier_hmcode_noradiation_growth_compute(pfo, pnw),
             pnw->pfg->error_message,
             pfo->error_message);

  return _SUCCESS_;
}

int fourier_hmcode_noradiation_growth_free(struct fourier* pfo, struct fourier_workspace* pnw){
  struct fourier_growth* pfg;
  pfg = pnw->pfg;
  free(pfg->a_table);
  free(pfg->growth_table);
  free(pfg->normgrowth_table);
  free(pfg->ddnormgrowth_table);
  free(pfg->dda_table);
  free(pfg->pvecnorad);
  free(pfg);
  return _SUCCESS_;
}

int fourier_hmcode_noradiation_growth_compute( struct fourier* pfo, struct fourier_workspace* pnw){

  struct fourier_growth* pfg;
  int i;
  double* pvec_growth;
  int* used_in_output;
  int (*generic_evolver)() = evolver_ndf15;
  pfg = pnw->pfg;

  class_alloc(pvec_growth,pfg->fg_size*sizeof(double),pfg->error_message);
  class_alloc(used_in_output,pfg->fg_size*sizeof(int),pfg->error_message);
  for(i=0;i<pfg->fg_size;++i){
    used_in_output[i] = _TRUE_;
  }

  class_call(fourier_hmcode_norad(pfg, pfg->a_ini),
             pfg->error_message,
             pfg->error_message);
  double f = 1.-pfg->pvecnorad[pfg->index_norad_Om];

  pvec_growth[pfg->index_fg_g] = pow(pfg->a_ini,1.-3.*f/5.);
  pvec_growth[pfg->index_fg_dg] = (1.-3.*f/5.)*pow(pfg->a_ini,-3.*f/5.);

  /** - perform the integration */
  double tol_fourier_growth_integration = 1e-10;
  class_call(generic_evolver(fourier_hmcode_growth_derivs,
                             pfg->a_ini,
                             pfg->a_final,
                             pvec_growth,
                             used_in_output,
                             pfg->fg_size,
                             pfg,
                             tol_fourier_growth_integration,
                             pfg->smallest_allowed_variation,
                             NULL,
                             1.,
                             pfg->a_table,
                             pfg->a_size,
                             fourier_hmcode_growth_sources,
                             NULL,
                             pfg->error_message),
             pfg->error_message,
             pfg->error_message);

  class_call(array_spline_table_line_to_line(
				          pfg->a_table,
				          pfg->a_size,
				          pfg->growth_table,
				          pfg->gt_size,
				          pfg->index_gt_g,
				          pfg->index_gt_ddg,
				          _SPLINE_EST_DERIV_,
				          pfg->error_message),
				    pfg->error_message,
				    pfg->error_message);
  class_call(array_integrate_spline_table_line_to_line(
					        pfg->a_table,
					        pfg->a_size,
					        pfg->growth_table,
					        pfg->gt_size,
					        pfg->index_gt_g,
					        pfg->index_gt_ddg,
					        pfg->index_gt_intg,
					        pfg->error_message),
					   pfg->error_message,
					   pfg->error_message);
  class_call(array_spline_table_line_to_line(
				          pfg->a_table,
				          pfg->a_size,
				          pfg->growth_table,
				          pfg->gt_size,
				          pfg->index_gt_intg,
				          pfg->index_gt_ddintg,
				          _SPLINE_EST_DERIV_,
				          pfg->error_message),
				    pfg->error_message,
				    pfg->error_message);
  class_call(array_spline_table_line_to_line(
				          pfg->a_table,
				          pfg->a_size,
				          pfg->growth_table,
				          pfg->gt_size,
				          pfg->index_gt_Omnorad,
				          pfg->index_gt_ddOmnorad,
				          _SPLINE_EST_DERIV_,
				          pfg->error_message),
				    pfg->error_message,
				    pfg->error_message);
  for(i=0;i<pfg->a_size;++i){
    pfg->normgrowth_table[i] /= pfg->normgrowth_table[pfg->a_size-1];
    //printf("%i - %.10e %.10e \n",i,pfg->a_table[i],pfg->normgrowth_table[i]);
  }
  class_call(array_spline_table_columns(
				          pfg->a_table,
				          pfg->a_size,
				          pfg->normgrowth_table,
				          1,
				          pfg->ddnormgrowth_table,
				          _SPLINE_EST_DERIV_,
				          pfg->error_message),
				    pfg->error_message,
				    pfg->error_message);
  class_call(array_spline_table_columns(
				          pfg->normgrowth_table,
				          pfg->a_size,
				          pfg->a_table,
				          1,
				          pfg->dda_table,
				          _SPLINE_EST_DERIV_,
				          pfg->error_message),
				    pfg->error_message,
				    pfg->error_message);
  for(i=0;i<pfg->a_size;++i){
    //printf("%i - %.10e %.10e %.10e \n",i,pfg->a_table[i],pfg->ddnormgrowth_table[i],pfg->dda_table[i]);
  }

  free(pvec_growth);
  free(used_in_output);
  return _SUCCESS_;
}

int fourier_hmcode_growth_derivs(
                      double a,
                      double* y,
                      double* dy,
                      void * parameters_and_workspace,
                      ErrorMsg error_message
                      ){
  struct fourier_growth * pfg;
  double Om_norad;
  double AH_norad, H2_norad;

  pfg = parameters_and_workspace;

  class_call(fourier_hmcode_norad(pfg, a),
             pfg->error_message,
             pfg->error_message);

  H2_norad = pfg->pvecnorad[pfg->index_norad_H2];
  AH_norad = pfg->pvecnorad[pfg->index_norad_AH];
  Om_norad = pfg->pvecnorad[pfg->index_norad_Om];

  dy[pfg->index_fg_g] = y[pfg->index_fg_dg];
  dy[pfg->index_fg_dg] = 1.5*Om_norad*y[pfg->index_fg_g]/(a*a) - (2.+AH_norad/H2_norad)*y[pfg->index_fg_dg]/a;

  return _SUCCESS_;
}

int fourier_hmcode_growth_sources(
                       double a,
                       double * y,
                       double * dy,
                       int index_a,
                       void * parameters_and_workspace,
                       ErrorMsg error_message
                       ) {
  struct fourier_growth * pfg;

  pfg = parameters_and_workspace;

  pfg->a_table[index_a] = a;
  pfg->growth_table[index_a*pfg->gt_size+pfg->index_gt_g] = y[pfg->index_fg_g]/a; //It is actually g(a)/a
  pfg->normgrowth_table[index_a] = y[pfg->index_fg_g];
  class_call(fourier_hmcode_norad(pfg, a),
             pfg->error_message,
             pfg->error_message);

  pfg->growth_table[index_a*pfg->gt_size+pfg->index_gt_Omnorad] = pfg->pvecnorad[pfg->index_norad_Om];

  return _SUCCESS_;
}

int fourier_hmcode_norad(struct fourier_growth* pfg, double a){

  double AHnr_nu, H2nr_nu;
  double w_v = -1.;
  double w_de = pfg->w0+(1.-a)*pfg->wa;
  double X_de = pow(a, -3.*(1.+pfg->w0+pfg->wa))*exp(-3.*pfg->wa*(1.-a));
  double Om_norad;
  double AH_norad, H2_norad;
  double a3;
  a3 = a*a*a;

  if( a > pfg->a_nu ){
    AHnr_nu = pfg->Omega_nu/a3;
    H2nr_nu = pfg->Omega_nu/a3;
  }
  else{
    H2nr_nu = 0.;
    AHnr_nu = 0.;
  }

  AH_norad = -(pfg->Omega_cdm/a3+pfg->Omega_b/a3+AHnr_nu+pfg->Omega_v*(1.+3.*w_v) + pfg->Omega_w*(1.+3.*w_de)*X_de)/2.;
  H2_norad = (pfg->Omega_cdm/a3 + pfg->Omega_b/a3 + H2nr_nu + pfg->Omega_v + pfg->Omega_w*X_de + (1.-pfg->Omega_m-pfg->Omega_v-pfg->Omega_w)/a/a);

  H2_norad+=pfg->Omega_nu*(pow(1.+pow(0.3173*a/pfg->a_nu,1.83),1./1.83)-1.)/a3/a;//Correction from inconsistent modeling between X_nu and H2_norad in HMcode

  if( a > pfg->a_nu ){
    Om_norad = pfg->Omega_m/a3/H2_norad;
  }
  else{
    Om_norad = (pfg->Omega_cdm + pfg->Omega_b)/a3/H2_norad;
  }
  pfg->pvecnorad[pfg->index_norad_H2] = H2_norad;
  pfg->pvecnorad[pfg->index_norad_AH] = AH_norad;
  pfg->pvecnorad[pfg->index_norad_Om] = Om_norad;

  return _SUCCESS_;
}

double fourier_hmcode_cbnu_ratio(double k, double z, double fnu, double om_m, double Tcmb, double growth){

  double pcb=(5.-sqrt(1+24*(1.-fnu)))/4;

  double BigT=Tcmb/2.7;

  double zeq=(2.5e4)*om_m*pow(BigT,-4.);
  //IF (EdS_Tcold_growth) THEN
  //    D=(1.+zeq)/(1.+z) ! EdS solution
  //ELSE
  //    D=(1.+zeq)*ungrow(z, cosm) ! General solution
  //END IF
  double D=(1.+zeq)*growth;//(1.+zeq)/(1.+z);

  double q=k*BigT*BigT/om_m;

  double Nnu = 1;
  double yfs=17.2*fnu*(1.+0.488*pow(fnu,-7./6.))*pow(Nnu*q/fnu,2);

  double Dcb=pow(1.+pow(D/(1.+yfs),0.7),pcb/0.7);
  double Dcbnu=pow(pow(1-fnu,0.7/pcb)+pow(D/(1.+yfs),0.7),pcb/0.7);

  return Dcb/Dcbnu;

}
