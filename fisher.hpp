#ifndef FISHER_HPP_
#define FISHER_HPP_
#include "TaylorF2e.hpp"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <gsl/gsl_randist.h>



double inner_product(vector<complex<double>> &h1, vector<complex<double>> &h2, vector<double> &noise, double df);
vector<complex<double>> sum_f2(vector<vector<complex<double>>> &vect);
vector<complex<double>> finite_diff(vector<complex<double>> &vect_right, vector<complex<double>> &vect_left, double ep);
vector<complex<double>> gen_waveform(double M, double eta, double e0, double p0, double A, double f0, double fend, double df);
vector<vector<complex<double>>> gen_waveform_full(double M, double eta, double e0, double p0, double A, double f0, double fend, double df);
Eigen::MatrixXd fim(vector<double> &loc, vector<double> &noise, double df, double f0, double fend, double ep);
Eigen::MatrixXd fim(vector<double> &loc, vector<double> &noise, double df, double f0, double fend, double ep, double T);
void fisher_prop(vector<double> &loc, vector<double> &prop, Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es, const gsl_rng * r);
vector<vector<complex<double>>> gen_waveform_full(double M, double eta, double e0, double p0, double A, double f0, double fend, double df);
double prod_rev(vector<double> &Ai, vector<double> &Aj, vector<double> &A, vector<double> &Phii, vector<double> &Phij, vector<double> &noise, double &df);
double prod_rev(vector<vector<double>> &deriv_i, vector<vector<double>> &deriv_j, vector<vector<double>> &Amps, vector<double> &noise, double &df);
vector<vector<double>> get_amp_phs(vector<vector<complex<double>>> &harm_wav);
vector<vector<double>> gen_amp_phs(double M, double eta, double e0, double p0, double A, double f0, double fend, double df);
vector<double> finite_diff(vector<double> &vect_right, vector<double> &vect_left, double ep);
vector<vector<double>> finite_diff(vector<vector<double>> &vect_right, vector<vector<double>> &vect_left, double ep);
Eigen::MatrixXd fim(vector<double> &loc, vector<double> &noise, double f0, double fend, double df, double ep ,int i);
Eigen::MatrixXd fim(vector<double> &loc, vector<double> &noise, double f0, double fend, double df, double ep , double T, int i);
void fisher_prop_circ(vector<double> &loc, vector<double> &prop, Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es, const gsl_rng * r);
void fisher_prop_ecc(vector<double> &loc, vector<double> &prop, Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es, const gsl_rng * r);
Eigen::MatrixXd fim_ecc(vector<double> &loc, vector<double> &noise, double f0, double fend, double df, double ep ,int i);
#endif
