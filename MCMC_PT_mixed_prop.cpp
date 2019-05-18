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

// g++ -I/Users/blakemoore/eigen -I/Users/blakemoore/fftw/include -I/Users/blakemoore/Desktop/Grad\ Research/MCMC_ecc -I/Users/blakemoore/gsl/include -O3 -ffast-math -g3 -c -fmessage-length=0 -std=c++14 -MMD -MP -MF"MCMC_PT_mixed_prop.d" -MT"MCMC_PT_mixed_prop.o" -o "MCMC_PT_mixed_prop.o" "MCMC_PT_mixed_prop.cpp"
//build with
// g++ -L/Users/blakemoore/gsl/lib -L/Users/blakemoore/fftw/lib -o "tstmc_v2"  ../Amps.o ../TaylorF2e.o ../llhood_maxd.o ../fisher.o ./MCMC_PT_mixed_prop.o  -lfftw3 -lgsl -lgslcblas -lm

using namespace std;
using namespace Eigen;

string path_inj = "/mnt/lustrefs/work/blake.moore2/MCMC_ecc/injects/";
string path_noise = "/mnt/lustrefs/work/blake.moore2/MCMC_ecc/noise/";
string path_samples = "/mnt/lustrefs/work/blake.moore2/MCMC_ecc/full_mixed_prop/samples/";

//string path_inj = "/Users/blakemoore/Desktop/Grad Research/MCMC_ecc/injects/";
//string path_noise = "/Users/blakemoore/Desktop/Grad Research/MCMC_ecc/noise/";
//string path_samples = "/Users/blakemoore/Desktop/Grad Research/MCMC_ecc/full_mixed_prop/samples_DB/";

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
    vector<vector<double>> DE_samples;
    int DE_track;
    chain(double M, double eta, double e0, double p0, double A, double T, double fend, double df, double df_fish, double ep_fish, vector<double> &noise, vector<double> &noise_fish, vector<complex<double>> &h2) : temp(T), count_in_temp(0), count_swap(0), loglike_prop(0), hast_ratio(0), urv(0)
    {
        DE_track = 0;
        DE_samples.resize(1000, vector<double> (5));
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

void cout_chain_info(chain c1);
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
void jump(vector<double> &loc, vector<double> &prop, SelfAdjointEigenSolver<Eigen::MatrixXd> &es, const gsl_rng * r, double &likehere, double &likeprop, double &rh, double T, double fend, double df, vector<complex<double>> &h2, vector<double> &noise, int &counter, int N_DE_samples, vector<vector<double>> &DE_samples);
void jump(chain &c, const gsl_rng * r, double fend, double df, vector<complex<double>> &h2, vector<double> &noise);
void inter_chain_swap(vector<double> &loc1, vector<double> &loc2, double &likehere1, double &likehere2, const gsl_rng * r, double T1, double T2, double fend, double df, vector<complex<double>> &h2, vector<double> &noise, MatrixXd &fish1, MatrixXd &fish2, SelfAdjointEigenSolver<Eigen::MatrixXd> &es1 , SelfAdjointEigenSolver<Eigen::MatrixXd> &es2, int &counter);
void swap_fishers(MatrixXd &fish1, MatrixXd &fish2, SelfAdjointEigenSolver<Eigen::MatrixXd> &es1 , SelfAdjointEigenSolver<Eigen::MatrixXd> &es2, double T1, double T2);
void inter_chain_swap(chain &c1, chain &c2, const gsl_rng * r, double fend, double df, vector<complex<double>> &h2, vector<double> &noise);
void record(chain &c, vector<vector<vector<double>>> &chain_store, vector<vector<double>> &like_store, int chain_num, int i);
void update_fisher(vector<double> &loc, MatrixXd &fish, SelfAdjointEigenSolver<Eigen::MatrixXd> &es, double T, double fend, double df, vector<double> &noise_fish, double ep_fish);
void update_fisher(chain &c, double fend, double df, vector<double> &noise_fish, double ep_fish);
void write_to_DE(chain &c);
void DE_prop(vector<double> &loc, vector<double> &prop, const gsl_rng * r, int N_DE_samples, vector<vector<double>> &DE_samples);

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
    int N_chain = stoi(argv[8]);
    double spacing = stod(argv[9]);
    vector<chain> chains;
    
    //initialize many chains
    for(int i = 0; i < N_chain; i++){
        chain c(M_in, eta_in, e0_in, p0_in, A_in, 1*pow(spacing, i), fend, df, df_fish, ep_fish, noise, noise_fish, h2);
        chains.push_back(c);
    }
    
    ////////////////////////////////////////////////////////
    // MCMC Routine
    ////////////////////////////////////////////////////////
    
    int N_jumps = stoi(argv[1]);
    const gsl_rng * r = gsl_rng_alloc (gsl_rng_taus);
    int within_temp = 0;
    int inter_chain = 0;
    vector<vector<vector<double>>> chain_store(N_chain, vector<vector<double>>(N_jumps, vector<double>(5)));
    vector<vector<double>> like_store(N_chain, vector<double>(N_jumps));
    for(int i = 0; i < N_jumps; i++){
        if( i % 5 != 0){ //Within Tempurature jumps
            for(int j = 0; j < N_chain; j++){
                jump(chains[j], r, fend, df, h2 ,noise);
            }
            within_temp++;
        } else { //Interchain jumps
            for(int j = 0; j < N_chain - 1; j++){
                inter_chain_swap(chains[j], chains[j+1], r, fend, df, h2, noise);
            }
            inter_chain++;
        }
        
        //Record Likelihood and Samples.
        for(int j = 0; j < N_chain; j++){
            record(chains[j], chain_store, like_store, j, i);
        }
        //Write to differential evolution list every 100 jumps
        if ( i % 100 == 0){
            for(int j = 0; j < N_chain; j++){
                write_to_DE(chains[j]);
            }
        }
        //Periodically update Fishers.
        if( i % 800 == 0 ){
            for(int j = 0; j < N_chain; j++){
                update_fisher(chains[j], fend, df_fish, noise_fish, ep_fish);
            }
        }
        //Periodically print acceptance ratios
        if ( i % 10000 == 0){
            cout << "within temp = " << within_temp << endl;
            for(int j = 0; j < N_chain; j++){
                cout << "Chain " << j << " temp = " << chains[j].temp << ", within temp accpt = " << (double) chains[j].count_in_temp/within_temp
                << ", inter chain accpt = " << (double) chains[j].count_swap/inter_chain << endl;
            }
            cout << endl;
        }
        //Periodically write data to file
        if ( i % 20000 == 0){
            for(int j = 0; j < N_chain; j++){
                write_vec_to_file(chain_store[j], "Samples_N_"+to_string(N_jumps)+"_chain_"+to_string(j)+"_inj"+to_string(stoi(argv[2]))+"_c_"+to_string(spacing)+".txt", path_samples);
                write_vec_to_file(like_store[j], "likelihood_N_"+to_string(N_jumps)+"_chain_"+to_string(j)+"_inj"+to_string(stoi(argv[2]))+"_c_"+to_string(spacing)+".txt", path_samples);
            }
        }
    }
    
    for(int j = 0; j < N_chain; j++){
        cout << "Chain " << j << " temp = " << chains[j].temp << ", within temp accpt = " << (double) chains[j].count_in_temp/within_temp
        << ", inter chain accpt = " << (double) chains[j].count_swap/inter_chain << endl;
    }
    cout << endl;
    
    //Write samples to file
    for(int j = 0; j < N_chain; j++){
        write_vec_to_file(chain_store[j], "Samples_N_"+to_string(N_jumps)+"_chain_"+to_string(j)+"_inj"+to_string(stoi(argv[2]))+"_c_"+to_string(spacing)+".txt", path_samples);
        write_vec_to_file(like_store[j], "likelihood_N_"+to_string(N_jumps)+"_chain_"+to_string(j)+"_inj"+to_string(stoi(argv[2]))+"_c_"+to_string(spacing)+".txt", path_samples);
    }
    
    return 0;
}
void write_to_DE(chain &c){
    if(c.DE_track == 1000){
        c.DE_track = 0;
    }
    
    c.DE_samples[c.DE_track][0] = c.loc[0];
    c.DE_samples[c.DE_track][1] = c.loc[1];
    c.DE_samples[c.DE_track][2] = c.loc[2];
    c.DE_samples[c.DE_track][3] = c.loc[3];
    c.DE_samples[c.DE_track][4] = c.loc[4];
    c.DE_track++;
}
void DE_prop(vector<double> &loc, vector<double> &prop, const gsl_rng * r, int N_DE_samples, vector<vector<double>> &DE_samples){
    double fact = gsl_ran_gaussian (r, 0.751319);
    int i = floor(gsl_ran_flat(r, 0, N_DE_samples - 1));
    int j = floor(gsl_ran_flat(r, 0, N_DE_samples - 1));
    prop[0] = exp(log(loc[0]) + fact*(log(DE_samples[i][0]) - log(DE_samples[j][0])));
    prop[1] = loc[1] + fact*(DE_samples[i][1] - DE_samples[j][1]);
    prop[2] = loc[2] + fact*(DE_samples[i][2] - DE_samples[j][2]);
    prop[3] = loc[3] + fact*(DE_samples[i][3] - DE_samples[j][3]);
    prop[4] = exp(log(loc[4]) + fact*(log(DE_samples[i][4]) - log(DE_samples[j][4])));
}
void update_fisher(chain &c, double fend, double df, vector<double> &noise_fish, double ep_fish){
    update_fisher(c.loc, c.fisher, c.eigen_sys, c.temp, fend, df, noise_fish, ep_fish);
}
void update_fisher(vector<double> &loc, MatrixXd &fish, SelfAdjointEigenSolver<Eigen::MatrixXd> &es, double T, double fend, double df, vector<double> &noise_fish, double ep_fish){
    fish = fim(loc, noise_fish, 0, fend, df, ep_fish, T, 3);
    es.compute(fish);
}
void record(chain &c, vector<vector<vector<double>>> &chain_store, vector<vector<double>> &like_store, int chain_num, int i){
    write_vec_to_vec(chain_store[chain_num], c.loc, i);
    like_store[chain_num][i] = c.loglike_loc;
}
void jump(chain &c, const gsl_rng * r, double fend, double df, vector<complex<double>> &h2, vector<double> &noise){
    jump(c.loc, c.prop, c.eigen_sys, r, c.loglike_loc, c.loglike_prop, c.hast_ratio, c.temp, fend, df, h2, noise, c.count_in_temp, c.DE_track, c.DE_samples);
}
void jump(vector<double> &loc, vector<double> &prop, SelfAdjointEigenSolver<Eigen::MatrixXd> &es, const gsl_rng * r, double &likehere, double &likeprop, double &rh, double T, double fend, double df, vector<complex<double>> &h2, vector<double> &noise, int &counter, int N_DE_samples, vector<vector<double>> &DE_samples){
    double jump_roll = gsl_ran_flat(r, 0, 1);
    if(jump_roll > 0.2){
        fisher_prop(loc, prop, es, r);
    } else if (jump_roll > 0.05){
        if(N_DE_samples < 2) {fisher_prop(loc, prop, es, r);}
        else {DE_prop(loc, prop, r, N_DE_samples, DE_samples);}
    } else {
        prior_prop(prop, r);
    }
    int check_pri = check_priors(prop);
    if(check_pri != 1){
        double urn = gsl_ran_flat(r, 0, 1.);
        likeprop = loglike(prop, 0, fend, df, h2, noise, T);
        rh = min(1., exp(likeprop - likehere));
        if(rh >= urn){
            set_loc(prop, loc);
            likehere = likeprop;
            counter++;
        }
    }
}
void inter_chain_swap(chain &c1, chain &c2, const gsl_rng * r, double fend, double df, vector<complex<double>> &h2, vector<double> &noise){
    inter_chain_swap(c1.loc, c2.loc, c1.loglike_loc, c2.loglike_loc, r, c1.temp, c2.temp, fend, df, h2, noise, c1.fisher, c2.fisher, c1.eigen_sys, c2.eigen_sys, c1.count_swap);
}
void inter_chain_swap(vector<double> &loc1, vector<double> &loc2, double &likehere1, double &likehere2, const gsl_rng * r, double T1, double T2, double fend, double df, vector<complex<double>> &h2, vector<double> &noise, MatrixXd &fish1, MatrixXd &fish2, SelfAdjointEigenSolver<Eigen::MatrixXd> &es1 , SelfAdjointEigenSolver<Eigen::MatrixXd> &es2, int &counter){
    double likeT1X2 = loglike(loc2, 0, fend, df, h2, noise, T1);
    double likeT2X1 = loglike(loc1, 0, fend, df, h2, noise, T2);
    double rh = min(1., exp((likeT1X2 + likeT2X1)-(likehere1 + likehere2)));
    double urn = gsl_ran_flat(r, 0, 1.);
    
    if(rh >= urn){
        swap_loc(loc1, loc2);
        swap_fishers(fish1, fish2, es1, es2, T1, T2);
        likehere1 = likeT1X2;
        likehere2 = likeT2X1;
        counter++;
    }
}
void swap_fishers(MatrixXd &fish1, MatrixXd &fish2, SelfAdjointEigenSolver<Eigen::MatrixXd> &es1 , SelfAdjointEigenSolver<Eigen::MatrixXd> &es2, double T1, double T2){
    MatrixXd fish_tmp;
    fish_tmp = fish1;
    fish1 = fish2*T2/T1;
    fish2 = fish_tmp*T1/T2;
    es1.compute(fish1);
    es2.compute(fish2);
}
void cout_chain_info(chain c1){
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
    cout << c1.fisher << endl;
    cout << "Eigensys" << endl;
    cout << c1.eigen_sys.eigenvalues() << endl;
    cout << c1.eigen_sys.eigenvectors() << endl;
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
        out << ' ' << vect[i][0] << ' ' << vect[i][1] << ' ' << vect[i][2] << ' ' << vect[i][3] << ' ' << vect[i][4] << endl;
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

