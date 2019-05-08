#include <vector>
#include <iostream>
#include <math.h>
#include <string>
#include <iomanip>
#include <stdio.h>
#include <fstream>
#include <ostream>
#include <gsl/gsl_randist.h>
#include <llhood_maxd.hpp>
#include <fisher.hpp>
#include <time.h>

// g++ -I/Users/blakemoore/eigen -I/Users/blakemoore/fftw/include -I/Users/blakemoore/Desktop/Grad\ Research/MCMC_ecc -I/Users/blakemoore/gsl/include -O3 -ffast-math -g3 -c -fmessage-length=0 -std=c++14 -MMD -MP -MF"MCMC_PT_v2.d" -MT"MCMC_PT_v2.o" -o "MCMC_PT_v2.o" "MCMC_PT_v2.cpp"
//build with
// g++ -L/Users/blakemoore/gsl/lib -L/Users/blakemoore/fftw/lib -o "tstmc_v2"  ../Amps.o ../TaylorF2e.o ../llhood_maxd.o ../fisher.o ./MCMC_PT_v2.o  -lfftw3 -lgsl -lgslcblas -lm

using namespace std;
using namespace Eigen;

//string path_inj = "/mnt/lustrefs/work/blake.moore2/MCMC_ecc/injects/";
//string path_noise = "/mnt/lustrefs/work/blake.moore2/MCMC_ecc/noise/";
//string path_samples = "/mnt/lustrefs/work/blake.moore2/MCMC_ecc/fisher_PT/samples/";

string path_inj = "/Users/blakemoore/Desktop/Grad Research/MCMC_ecc/injects/";
string path_noise = "/Users/blakemoore/Desktop/Grad Research/MCMC_ecc/noise/";
string path_samples = "/Users/blakemoore/Desktop/Grad Research/MCMC_ecc/fisher_prop_PT/samples_DB/";

struct chain{
    double temp;
    double loglike_loc;
    double loglike_prop;
    double hast_ratio;
    double urv;
    int count_in_temp;
    int count_swap;
    vector<double> loc;
    vector<double> prop;
    MatrixXd fisher;
    SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_sys;
    chain(double M, double eta, double e0, double p0, double A, double T, double fend, double df, double df_fish, double ep_fish, vector<double> &noise, vector<double> &noise_fish, vector<complex<double>> &h2) : temp(T), count_in_temp(0), count_swap(0), loglike_prop(0), hast_ratio(0), urv(0)
    {
        loc.resize(5);
        prop.resize(5);
        loc[0] = M;
        loc[1] = eta;
        loc[2] = e0;
        loc[3] = p0;
        loc[4] = A;
        loglike_loc = loglike(loc, 0, fend, df, h2, noise, T);
        fisher = fim(loc, noise_fish, 0, fend, df_fish, ep_fish, T, 3);
        eigen_sys.compute(fisher);
    }
};

int check_priors(vector<double> &prop); // will return 1 when out of prior bounds
void set_loc(vector<double> &prop, vector<double> &loc);
void write_vec_to_vec(vector<vector<double>> &samples, vector<double> &sample, int i);
void write_vec_to_file(vector<vector<double>> &samples, string filename, string path);
void write_vec_to_file(vector<complex<double>> &vect, string str, string path);
void write_vec_to_file(vector<double> &vect, string str, string path);
void cout_vec(vector<double> &vec);
vector<vector<double>> write_in_F2(string str, string path, double flim);
void prior_prop(vector<double> &prop, const gsl_rng * r);
void swap_loc(vector<double> &loc1, vector<double> &loc2);

int main(int argc, const char * argv[]){
    ////////////////////////////////////////////////////////
    // Write in data file (Injection)
    ////////////////////////////////////////////////////////
    
    vector<vector<double>> data = write_in_F2("Inject_"+to_string(stoi(argv[2]))+".txt", path_inj, 1000);
    int data_size = data.size();
    vector<double> freqs(data_size);
    vector<complex<double>> h2(data_size);
    for(int i = 0; i < data_size; i++){
        freqs[i] = data[i][0];
        h2[i] = data[i][1] + 1i*data[i][2];
    }
    double df = freqs[1];
    double fend = freqs[data_size - 1];
    
    ////////////////////////////////////////////////////////
    // Set up noise for likelihood evaluations
    // Set up downsampled noise for fisher
    ////////////////////////////////////////////////////////
    double x[3000];
    double y[3000];
    int j = 0;
    double in1;
    double in2;
    int n = 3000;
    
    ifstream noisedat2 (path_noise + "/AdLIGODwyer.dat");
    while(noisedat2 >> in1 >> in2){
        x[j] = in1;
        y[j] = log(in2*in2);
        j++;
    }
    
    gsl_interp_accel *acc
    = gsl_interp_accel_alloc ();
    gsl_spline *spline
    = gsl_spline_alloc (gsl_interp_cspline, n);
    gsl_spline_init (spline, x, y, n);
    
    //noise for the likelihood
    vector<double> noise(data_size);
    for (int i = 0; i < data_size; i++)    {
        if (freqs[i] > 1 &&  freqs[i] < 4096 ) {
            noise[i] = gsl_spline_eval (spline, freqs[i], acc);
        } else {
            noise[i] = pow(10,10); //effectively make the noise infinite below 1Hz
        }
    }
    
    //downsampled noise for the fisher
    double df_fish = 0.25;
    double ep_fish = 1e-8;
    int N_down_noise = fend/df_fish + 1;
    vector<double> noise_fish(N_down_noise);
    double f = 0;
    for (int i = 0; i < N_down_noise; i++)    {
        f = df_fish*i;
        if (f > 1 &&  f < 4096 ) {
            noise_fish[i] = gsl_spline_eval (spline, f, acc);
        } else {
            noise_fish[i] = pow(10,10); //effectively make the noise infinite below 1Hz
        }
    }
    
    ////////////////////////////////////////////////////////
    // Initialize the chains
    ////////////////////////////////////////////////////////
    
    double M_in = stod(argv[3]);
    double eta_in = stod(argv[4]);
    double e0_in = stod(argv[5]);
    double p0_in = stod(argv[6]);
    double A_in =  exp(stod(argv[7]));
    
    chain c1(M_in, eta_in, e0_in, p0_in, A_in, 1, fend, df, df_fish, ep_fish, noise, noise_fish, h2);
    
    cout << "Temp = " << c1.temp << endl;
    cout << "like_loc = " << c1.loglike_loc << endl;
    cout << "like_prop = " << c1.loglike_prop << endl;
    cout << "hast = " << c1.hast_ratio << endl;
    cout << "urv = " << c1.urv << endl;
    cout << "count temp = " << c1.count_in_temp << endl;
    cout << "count temp = " << c1.count_swap << endl;
    cout << "location" << endl;
    cout_vec(c1.loc);
    cout << "proposal" << endl;
    cout_vec(c1.prop);
    cout << "Fisher" << endl;
    cout << c1.fisher;
    cout << "Eigensys" << endl;
    cout << c1.eigen_sys.eigenvalues() << endl;
    cout << c1.eigen_sys.eigenvectors() << endl;
    
    return 0;
}


int check_priors(vector<double> &prop){
    int cont = 0;
    if(prop[0] > 38 || prop[0] < 1) {cont = 1;}
    if(prop[1] > 0.25 || prop[1] < 0.15) {cont = 1;}
    if(prop[2] > 0.8 || prop[2] < 0.0001) {cont = 1;}
    if(prop[3] > 150 || prop[3] < 40) {cont = 1;}
    if(prop[4] > 1e-16 || prop[4] < 1e-20) {cont = 1;}
    return cont;
}
void prior_prop(vector<double> &prop, const gsl_rng * r){
    prop[0] = exp(gsl_ran_flat(r, 0, 3.63759));
    prop[1] = gsl_ran_flat(r, 0.15, 0.25);
    prop[2] = gsl_ran_flat(r, 0.0001, 0.8);
    prop[3] = gsl_ran_flat(r, 40, 150);
    prop[4] = exp(gsl_ran_flat(r, -46.0517, -36.8414));
}
void set_loc(vector<double> &prop, vector<double> &loc){
    loc[0] = prop[0];
    loc[1] = prop[1];
    loc[2] = prop[2];
    loc[3] = prop[3];
    loc[4] = prop[4];
}
void swap_loc(vector<double> &loc1, vector<double> &loc2){
    vector<double> tmp(5);
    set_loc(loc1, tmp);
    set_loc(loc2, loc1);
    set_loc(tmp, loc2);
}
void write_vec_to_vec(vector<vector<double>> &samples, vector<double> &sample, int i){
    samples[i][0] = sample[0];
    samples[i][1] = sample[1];
    samples[i][2] = sample[2];
    samples[i][3] = sample[3];
    samples[i][4] = sample[4];
}
void write_vec_to_file(vector<vector<double>> &vect, string filename, string path){
    ofstream out;
    out.open(path + filename);
    for (int i=0; i<vect.size(); i++){
        out << setprecision(16) << ' ' << vect[i][0] << ' ' << vect[i][1] << ' ' << vect[i][2] << ' ' << vect[i][3] << ' ' << vect[i][4] << endl;
    }
    out.close();
}

void cout_vec(vector<double> &vec){
    cout << setprecision(16) << " M = " << vec[0] <<  " eta = " << vec[1] <<  " e0 = " << vec[2] <<  " p0 = " << vec[3] <<  " amp = " << vec[4] << endl;
}
void write_vec_to_file(vector<complex<double>> &vect, string str, string path){
    ofstream out;
    out.open(path + str);
    for (int i=0; i<vect.size(); i++){
        out << setprecision(16) << ' ' << real(vect[i]) << ' ' << imag(vect[i]) << ' ' << endl;
    }
    out.close();
}
void write_vec_to_file(vector<double> &vect, string str, string path){
    ofstream out;
    out.open(path + str);
    for (int i=0; i<vect.size(); i++){
        out << setprecision(16) << ' ' << vect[i] << endl;
    }
    out.close();
}
vector<vector<double>> write_in_F2(string str, string path, double flim){
    vector<vector<double>> out(2, vector<double>(3));
    double re = 0;
    double im = 0;
    double df = 0.015625;
    int j = 0;
    int N = ceil(flim/df);
    out.resize(N, vector<double>(3));
    
    ifstream t4in (path + str);
    while(j < N){
        t4in >> re >> im;
        out[j][0] = (double) df*j;
        out[j][1] = re;
        out[j][2] = im;
        j++;
    }
    return out;
}

