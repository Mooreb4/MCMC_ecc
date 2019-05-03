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

using namespace std;
using namespace Eigen;

vector<vector<double>> write_in_T4(string str, string path, double flim);

int check_priors(vector<double> &prop); // will return 1 when out of prior bounds
void set_loc(vector<double> &prop, vector<double> &loc);
void write_vec_to_vec(vector<vector<double>> &samples, vector<double> &sample, int i);
void write_vec_to_file(vector<vector<double>> &samples, string filename, string path);
void write_vec_to_file(vector<complex<double>> &vect, string str, string path);
void write_vec_to_file(vector<double> &vect, string str, string path);
void cout_vec(vector<double> &vec);
vector<vector<double>> write_in_F2(string str, string path, double flim);

string path_inj = "/mnt/lustrefs/work/blake.moore2/MCMC_ecc/injects/";
string path_noise = "/mnt/lustrefs/work/blake.moore2/MCMC_ecc/noise/";
string path_samples = "/mnt/lustrefs/work/blake.moore2/MCMC_ecc/fisher_prop/samples/";

int main(int argc, const char * argv[]){
    
    // write in data and create list of frequencies and a vector<complex> for the injection
    vector<vector<double>> data = write_in_F2("Inject_"+to_string(stoi(argv[2]))+".txt", path_inj, 1000);
    cout << "read in" << endl;
    int data_size = data.size();
    vector<double> freqs(data_size);
    vector<complex<double>> h2(data_size);
    
    for(int i = 0; i < data_size; i++){
        freqs[i] = data[i][0];
        h2[i] = data[i][1] + 1i*data[i][2];
    }
    
    double df = freqs[1];
    double fend = freqs[data_size - 1];
    
    ////////////////////////////////
    // Set up the noise
    ////////////////////////////////
    
    double x[3000];
    double y[3000];
    int j = 0;
    double in1;
    double in2;
    int n = 3000;
    
    //  set up an interpolation of the noise curve
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
    
    // evaluate the interpolation (note that I'm working with log noise which I'll unlog when computing inner products(logging makes the interpolation behave))
    vector<double> noise(data_size);
    
    for (int i = 0; i < data_size; i++)    {
        if (freqs[i] > 1 &&  freqs[i] < 4096 ) {
            noise[i] = gsl_spline_eval (spline, freqs[i], acc);
        } else {
            noise[i] = pow(10,10); //effectively make the noise infinite below 1Hz
        }
    }
    
    ////////////////////////////////////////////
    // Set up required pieces for this basic MCMC
    ////////////////////////////////////////////
    
    const gsl_rng * r = gsl_rng_alloc (gsl_rng_taus);
    vector<double> loc(5);
    vector<double> prop(5);
    vector<double> tmp_loc(5);
    loc[0] = stod(argv[3]); //M
    loc[1] = stod(argv[4]); // eta
    loc[2] = stod(argv[5]); //e0
    loc[3] = stod(argv[6]); //p0
    loc[4] =  exp(stod(argv[7])); //Ampfactor (given in log)
    int Njumps = stoi(argv[1]);
    int check_pri;
    double rh;
    double urn;
    int accpt = 0;
    int counter = 0;
    int tracker = 0;
    
    vector<vector<double>> samples (Njumps, vector<double> (5));
    vector<double> likelihood(Njumps);
    write_vec_to_vec(samples, loc, 0);
    
    double likehere = loglike(loc, 0, fend, df, h2, noise);
    double likeprop = 0;
    likelihood[0] = likehere;
    
    MatrixXd fish = fim(loc, noise, 0, fend, df, 1e-6);
    SelfAdjointEigenSolver<Eigen::MatrixXd> es(5);
    es.compute(fish);
    
    // MCMC routine
    
    for(int i = 1; i < Njumps; i++){
        fisher_prop(loc, prop, es, r);
        check_pri = check_priors(prop);
//        cout << "The Location " << endl;
//        cout_vec(loc);
//        cout << "The Proposal " << endl;
//        cout_vec(prop);
//        cout << "Prior check = " << check_pri << endl;
        if(check_pri == 1){
            write_vec_to_vec(samples, loc, i);
        } else {
            urn = gsl_ran_flat(r, 0, 1.);
            likeprop = loglike(prop, 0, fend, df, h2, noise);
            rh = min(1., exp(likeprop - likehere));
//            cout << "Hasting ratio = " << rh << endl;;
//            cout << "Like here = " << likehere << endl;
//            cout << "Like prop = " << likeprop << endl;
            if(rh >= urn){
                set_loc(prop, loc);
                likehere = likeprop;
                write_vec_to_vec(samples, loc, i);
                accpt++;
            } else {
                write_vec_to_vec(samples, loc, i);
            }
        }
        
        //update fisher
        counter++;
        if(counter == 400){
            if(loc[2] > 0.1){
                fish = fim(loc, noise, 0, fend, df, 1e-6);
                es.compute(fish);
                counter = 0;}
            else {
                set_loc(loc, tmp_loc);
                tmp_loc[2] = 0.1;
                fish = fim(tmp_loc, noise, 0, fend, df, 1e-6);
                es.compute(fish);
                }
        }
        //write likelihood
        likelihood[i] = likehere;
        
        //tracking
        tracker++;
        if(tracker == 10000){
            cout << "The Location " << endl;
            cout_vec(loc);
            cout << "The Proposal " << endl;
            cout_vec(prop);
            cout << "fisher " << endl;
            cout << fish << endl;
            cout << "eigenvalues" << endl;
            cout << es.eigenvalues() << endl;
            cout << "eigenvectors" << endl;
            cout << es.eigenvectors() << endl;
	    cout << "hastings ratio = " << rh << endl;
            write_vec_to_file(samples, "Samples_N_"+to_string(Njumps)+"inj_"+to_string(stoi(argv[2]))+".txt", path_samples);
            write_vec_to_file(likelihood, "likelihood_N_"+to_string(Njumps)+"inj_"+to_string(stoi(argv[2]))+".txt", path_samples);
            tracker = 0;
        }
    }
    cout << " Acceptance ratio = " << (double) accpt/Njumps << endl;
    write_vec_to_file(samples, "Samples_N_"+to_string(Njumps)+"inj_"+to_string(stoi(argv[2]))+".txt", path_samples);
    write_vec_to_file(likelihood, "likelihood_N_"+to_string(Njumps)+"inj_"+to_string(stoi(argv[2]))+".txt", path_samples);
    
    
    return 0;
}

vector<vector<double>> write_in_T4(string str, string path, double flim){
    vector<vector<double>> out(2, vector<double>(3));
    double f = 0;
    double re = 0;
    double im = 0;
    double df = 0;
    int j = 0;
    
    ifstream t4in (path + str);
    t4in >> f >> re >> im;
    out[j][0] = f;
    out[j][1] = re;
    out[j][2] = im;
    j++;
    cout << "f = " << f << endl;
    t4in >> f >> re >> im;
    out[j][0] = f;
    out[j][1] = re;
    out[j][2] = im;
    j++;
    cout << "f = " << f << endl;
    df = out[1][0] - out[0][0];
    cout << "df = " << df << endl;
    
    int N = flim/df ;
    double log2N = log2(N);
    N = pow(2, ceil(log2N));
    out.resize(N, vector<double>(3));
    while(j < N){
        t4in >> f >> re >> im;
        out[j][0] = f;
        out[j][1] = re;
        out[j][2] = im;
        j++;
    }
    
    return out;
}
void gaussian_prop(vector<double> &prop, vector<double> &loc, double sig_M, double sig_eta, double sig_e0, double sig_p0, double sig_amp, const gsl_rng * r){
    prop[0] = loc[0] + gsl_ran_gaussian (r, sig_M);
    prop[1] = loc[1] + gsl_ran_gaussian (r, sig_eta);
    prop[2] = loc[2] + gsl_ran_gaussian (r, sig_e0);
    prop[3] = loc[3] + gsl_ran_gaussian (r, sig_p0);
    prop[4] = loc[4] + gsl_ran_gaussian (r, sig_amp);
}
int check_priors(vector<double> &prop){
    int cont = 0;
    if(prop[0] > 100 || prop[0] < 2) {cont = 1;}
    if(prop[1] > 0.25 || prop[1] < 0.01) {cont = 1;}
    if(prop[2] > 0.8 || prop[2] < 0.000001) {cont = 1;}
    if(prop[3] > 150 || prop[3] < 40) {cont = 1;}
    if(prop[4] > 1 || prop[4] < 0) {cont = 1;}
    return cont;
}
void set_loc(vector<double> &prop, vector<double> &loc){
    loc[0] = prop[0];
    loc[1] = prop[1];
    loc[2] = prop[2];
    loc[3] = prop[3];
    loc[4] = prop[4];
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
    cout << " M = " << vec[0] <<  " eta = " << vec[1] <<  " e0 = " << vec[2] <<  " p0 = " << vec[3] <<  " amp = " << vec[4] << endl;
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

