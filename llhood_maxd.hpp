#ifndef LLHOOD_MAXD_HPP_
#define LLHOOD_MAXD_HPP_
#include <fftw3.h>
#include <complex>
#include <vector>
#include <algorithm>
#include "TaylorF2e.hpp"

double get_snr_sq(vector<complex<double>> &vect, vector<double> &noise, double &df);
double get_snr_sq(vector<vector<complex<double>>> &vect, vector<double> &noise, double &df);
fftw_complex* take_invft_gj(vector<complex<double>> &h1, vector<complex<double>> &h2, vector<double> &noise);
vector<double> abs_gj(vector<complex<double>> &h1, vector<complex<double>> &h2, vector<double> &noise);
double inprod_h1_h2_maxd(vector<vector<complex<double>>> &h1, vector<complex<double>> &h2, vector<double> &noise, double &df);
double loglike(double M, double eta, double e0, double p0, double are, double aim, double f0, double fend, double df, vector<complex<double>> &h2, vector<double> &noise);
double loglike(double M, double eta, double e0, double p0, double ampmag, double f0, double fend, double df, vector<complex<double>> &h2, vector<double> &noise);
double loglike(vector<double> &params, double f0, double fend, double df, vector<complex<double>> &h2, vector<double> &noise);
double loglike(vector<double> &params, double f0, double fend, double df, vector<complex<double>> &h2, vector<double> &noise, double T);
#endif
