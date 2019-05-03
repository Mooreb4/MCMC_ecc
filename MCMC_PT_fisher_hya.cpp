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

// g++ -I/Users/blakemoore/eigen -I/Users/blakemoore/fftw/include -I/Users/blakemoore/Desktop/Grad\ Research/MCMC_ecc -I/Users/blakemoore/gsl/include -O3 -ffast-math -g3 -c -fmessage-length=0 -std=c++14 -MMD -MP -MF"MCMC_PT_fisher_hya.d" -MT"MCMC_PT_fisher_hya.o" -o "MCMC_PT_fisher_hya.o" "MCMC_PT_fisher_hya.cpp"
//build with
// g++ -L/Users/blakemoore/gsl/lib -L/Users/blakemoore/fftw/lib -o "tstmc"  ../Amps.o ../TaylorF2e.o ../llhood_maxd.o ../fisher.o ./MCMC_PT_fisher_hya.o  -lfftw3 -lgsl -lgslcblas -lm

using namespace std;
using namespace Eigen;

int check_priors(vector<double> &prop); // will return 1 when out of prior bounds
void set_loc(vector<double> &prop, vector<double> &loc);
void write_vec_to_vec(vector<vector<double>> &samples, vector<double> &sample, int i);
void write_vec_to_file(vector<vector<double>> &samples, string filename, string path);
void write_vec_to_file(vector<complex<double>> &vect, string str, string path);
void write_vec_to_file(vector<double> &vect, string str, string path);
void cout_vec(vector<double> &vec);
vector<vector<double>> write_in_F2(string str, string path, double flim);
void jump(vector<double> &loc, vector<double> &prop, SelfAdjointEigenSolver<Eigen::MatrixXd> &es, const gsl_rng * r, double &likehere, double &likeprop, double &rh, vector<vector<double>> &samples, double T, int i, double fend, double df, vector<complex<double>> &h2, vector<double> &noise, int &counter);
void update_fisher(vector<double> &loc, MatrixXd &fish, SelfAdjointEigenSolver<Eigen::MatrixXd> &es, double T, double fend, double df, vector<double> &noise);
void swap_loc(vector<double> &loc1, vector<double> &loc2);
void inter_chain_swap(vector<double> &loc1, vector<double> &loc2, double &likehere1, double &likehere2, const gsl_rng * r, vector<vector<double>> &samples1, vector<vector<double>> &samples2, double T1, double T2, int i, double fend, double df, vector<complex<double>> &h2, vector<double> &noise, MatrixXd &fish1, MatrixXd &fish2, SelfAdjointEigenSolver<Eigen::MatrixXd> &es1 , SelfAdjointEigenSolver<Eigen::MatrixXd> &es2, int &counter);
void swap_fishers(MatrixXd &fish1, MatrixXd &fish2, SelfAdjointEigenSolver<Eigen::MatrixXd> &es1 , SelfAdjointEigenSolver<Eigen::MatrixXd> &es2, double T1, double T2);

string path_inj = "/mnt/lustrefs/work/blake.moore2/MCMC_ecc/injects/";
string path_noise = "/mnt/lustrefs/work/blake.moore2/MCMC_ecc/noise/";
string path_samples = "/mnt/lustrefs/work/blake.moore2/MCMC_ecc/fisher_PT/samples_DB/";

//string path_inj = "/Users/blakemoore/Desktop/Grad Research/MCMC_ecc/injects/";
//string path_noise = "/Users/blakemoore/Desktop/Grad Research/MCMC_ecc/noise/";
//string path_samples = "/Users/blakemoore/Desktop/Grad Research/MCMC_ecc/fisher_prop_PT/samples_DB/";

int main(int argc, const char * argv[]){
    
    
    // write in data and create list of frequencies and a vector<complex> for the injection
    vector<vector<double>> data = write_in_F2("Inject_"+to_string(stoi(argv[2]))+".txt", path_inj, 1000);
    cout << "read in" << endl;
    int data_size = data.size();
    cout << "dat size = " << data_size << endl;
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
    cout << "noise done" << endl;
    
    ////////////////////////////////////////
    // Set up downsampled noise for the fisher
    ////////////////////////////////////////
    int N_down_noise = fend/0.1 + 1;
    vector<double> noise_fish(N_down_noise);
    double f = 0;
    for (int i = 0; i < N_down_noise; i++)    {
        f = 0.1*i;
        if (f > 1 &&  f < 4096 ) {
            noise_fish[i] = gsl_spline_eval (spline, f, acc);
        } else {
            noise_fish[i] = pow(10,10); //effectively make the noise infinite below 1Hz
        }
    }
    
    
    
    ////////////////////////////////////////
    // Initialize the ingredients for the different chains
    ////////////////////////////////////////
    int cont_chain = 0;
    int fish_counter = 0;
    
    
    const gsl_rng * r = gsl_rng_alloc (gsl_rng_taus);
    int Njumps = stoi(argv[1]);
    vector<double> loc1(5);
    vector<double> loc2(5);
    vector<double> loc3(5);
    vector<double> prop1(5);
    vector<double> prop2(5);
    vector<double> prop3(5);
    loc1[0] = stod(argv[3]); //M
    loc1[1] = stod(argv[4]); // eta
    loc1[2] = stod(argv[5]); //e0
    loc1[3] = stod(argv[6]); //p0
    loc1[4] =  exp(stod(argv[7])); //Ampfactor (given in log)
    set_loc(loc1, loc2);
    set_loc(loc1, loc3);
    double rh1;
    double rh2;
    double rh3;
    double urn1;
    double urn2;
    double urn3;
    
    // setting temps
    double c = stod(argv[8]);
    double T1 = 1;
    double T2 = T1*c;
    double T3 = T2*c;
    
    
    vector<vector<double>> samples1 (Njumps, vector<double> (5));
    vector<vector<double>> samples2 (Njumps, vector<double> (5));
    vector<vector<double>> samples3 (Njumps, vector<double> (5));
    vector<double> likelihood1(Njumps);
    vector<double> likelihood2(Njumps);
    vector<double> likelihood3(Njumps);
    write_vec_to_vec(samples1, loc1, 0);
    write_vec_to_vec(samples2, loc2, 0);
    write_vec_to_vec(samples3, loc3, 0);
    
    double likehere1 = loglike(loc1, 0, fend, df, h2, noise, T1);
    double likehere2 = loglike(loc2, 0, fend, df, h2, noise, T2);
    double likehere3 = loglike(loc3, 0, fend, df, h2, noise, T3);
    double likeprop1 = 0;
    double likeprop2 = 0;
    double likeprop3 = 0;
    likelihood1[0] = likehere1;
    likelihood2[0] = likehere2;
    likelihood3[0] = likehere3;
    
    
    MatrixXd fish1 = fim(loc1, noise_fish, 0, fend, 0.1, 1e-8, T1, 1);
    MatrixXd fish2 = fim(loc2, noise_fish, 0, fend, 0.1, 1e-8, T2, 1);
    MatrixXd fish3 = fim(loc3, noise_fish, 0, fend, 0.1, 1e-8, T3, 1);
    SelfAdjointEigenSolver<Eigen::MatrixXd> es1(5);
    SelfAdjointEigenSolver<Eigen::MatrixXd> es2(5);
    SelfAdjointEigenSolver<Eigen::MatrixXd> es3(5);
    es1.compute(fish1);
    es2.compute(fish2);
    es3.compute(fish3);
    
    //    cout_vec(loc1);
    //    cout_vec(loc2);
    //    cout_vec(loc3);
    cout << "Fisher updated to " << endl;
    cout << fish1 << endl;
    cout << endl;
    cout << "Eigen values are now " << endl;
    cout << endl;
    cout << es1.eigenvalues() << endl;
    cout << "Eigen Vectors are now " << endl;
    cout << endl;
    cout << es1.eigenvectors() << endl;
    cout << endl;
    //    cout << "Fisher 2 " << endl;
    //    cout << fish2 << endl;
    //    cout << "Fisher 3 " << endl;
    //    cout << fish3 << endl;
    
    int count_IC= 0;
    int count_IC_12=0;
    int count_IC_23=0;
    int count_1=0;
    int count_2=0;
    int count_3=0;
    int tracker = 0;
    
    ////////////////////////////////////////////////
    // The overall MCMC Routine now
    ////////////////////////////////////////////////
    
    for(int i = 1; i < Njumps; i++){
        if(cont_chain != 5){ //propose within chain jumps
            cont_chain++;
            
            // Cold chain
            jump(loc1, prop1, es1, r, likehere1, likeprop1, rh1, samples1, T1, i, fend, df, h2, noise, count_1);
            // T2
            jump(loc2, prop2, es2, r, likehere2, likeprop2, rh2, samples2, T2, i, fend, df, h2, noise, count_2);
//            // T3
            jump(loc3, prop3, es3, r, likehere3, likeprop3, rh3, samples3, T3, i, fend, df, h2, noise, count_3);
            
        } else {
            cont_chain = 0;
            //inter_chain_swaps
            inter_chain_swap(loc1, loc2, likehere1, likehere2, r, samples1, samples2, T1, T2, i, fend, df, h2, noise, fish1, fish2, es1, es2, count_IC_12);
            inter_chain_swap(loc2, loc3, likehere2, likehere3, r, samples2, samples3, T2, T3, i, fend, df, h2, noise, fish2, fish3, es2, es3, count_IC_23);
            count_IC++;
        }
        
        //Fisher updates
        if(fish_counter == 500){
            update_fisher(loc1, fish1, es1, T1, fend, .1, noise_fish);
            update_fisher(loc2, fish2, es2, T2, fend, .1, noise_fish);
            update_fisher(loc3, fish3, es3, T3, fend, .1, noise_fish);
            fish_counter = 0;
        }
        
        likelihood1[i] = likehere1;
        fish_counter++;
        
        //Tracking;
        tracker++;
        if(tracker == 10000){
            write_vec_to_file(samples1, "Samples_N_"+to_string(Njumps)+"_chain_1_inj"+to_string(stoi(argv[2]))+".txt", path_samples);
            write_vec_to_file(likelihood1, "likelihood_N_+"+to_string(Njumps)+"_chain_1_inj"+to_string(stoi(argv[2]))+".txt", path_samples);
            write_vec_to_file(samples2, "Samples_N_"+to_string(Njumps)+"_chain_2_inj"+to_string(stoi(argv[2]))+".txt", path_samples);
            write_vec_to_file(likelihood2, "likelihood_N_"+to_string(Njumps)+"_chain_2_inj"+to_string(stoi(argv[2]))+".txt", path_samples);
            write_vec_to_file(samples3, "Samples_N_"+to_string(Njumps)+"_chain_3_inj"+to_string(stoi(argv[2]))+".txt", path_samples);
            write_vec_to_file(likelihood3, "likelihood_N_"+to_string(Njumps)+"_chain_3_inj"+to_string(stoi(argv[2]))+".txt", path_samples);
        }
        
        
    }
    
    
    
    
    write_vec_to_file(samples1, "Samples_N_"+to_string(Njumps)+"_chain_1_inj"+to_string(stoi(argv[2]))+".txt", path_samples);
    write_vec_to_file(likelihood1, "likelihood_N_+"+to_string(Njumps)+"_chain_1_inj"+to_string(stoi(argv[2]))+".txt", path_samples);
    write_vec_to_file(samples2, "Samples_N_"+to_string(Njumps)+"_chain_2_inj"+to_string(stoi(argv[2]))+".txt", path_samples);
    write_vec_to_file(likelihood2, "likelihood_N_"+to_string(Njumps)+"_chain_2_inj"+to_string(stoi(argv[2]))+".txt", path_samples);
    write_vec_to_file(samples3, "Samples_N_"+to_string(Njumps)+"_chain_3_inj"+to_string(stoi(argv[2]))+".txt", path_samples);
    write_vec_to_file(likelihood3, "likelihood_N_"+to_string(Njumps)+"_chain_3_inj"+to_string(stoi(argv[2]))+".txt", path_samples);
//
    cout << "Interchain accpt 1 - 2 = " << (double) count_IC_12/count_IC << endl;
    cout << "Interchain accpt 2 - 3 = " << (double) count_IC_23/count_IC << endl;
    cout << "Chain 1 accpt ratio = " << (double) count_1/Njumps*6/5 << endl;
    cout << "Chain 2 accpt ratio = " << (double) count_2/Njumps*6/5 << endl;
    cout << "Chain 3 accpt ratio = " << (double) count_3/Njumps*6/5 << endl;
    
        cout << "Chain 1 accpt ratio = " << (double) count_1/Njumps << endl;
    
    
    return 0;
}

void jump(vector<double> &loc, vector<double> &prop, SelfAdjointEigenSolver<Eigen::MatrixXd> &es, const gsl_rng * r, double &likehere, double &likeprop, double &rh, vector<vector<double>> &samples, double T, int i, double fend, double df, vector<complex<double>> &h2, vector<double> &noise, int &counter){
    fisher_prop(loc, prop, es, r);
    int check_pri = check_priors(prop);
    
//        cout << "Chain temp = " << T << endl;
//        cout << "Loc =" << endl;
//        cout_vec(loc);
//        cout << "prop =" << endl;
//        cout_vec(prop);
//        cout << "like here = " << likehere << endl;
    
    if(check_pri == 1){
        write_vec_to_vec(samples, loc, i);
 //              cout << "out of prior range" << endl;
    } else {
        double urn = gsl_ran_flat(r, 0, 1.);
        likeprop = loglike(prop, 0, fend, df, h2, noise, T);
 //            cout << "prop like = " << likeprop << endl;
        rh = min(1., exp(likeprop - likehere));
  //            cout << "hastings ratio = " << rh << endl;
        if(rh >= urn){
            set_loc(prop, loc);
            likehere = likeprop;
            write_vec_to_vec(samples, loc, i);
  //                    cout << "accepted" << endl;
            counter++;
        } else {
            write_vec_to_vec(samples, loc, i);
   //                 cout << "rejected" << endl;
        }
    }
      cout << endl;
}
void inter_chain_swap(vector<double> &loc1, vector<double> &loc2, double &likehere1, double &likehere2, const gsl_rng * r, vector<vector<double>> &samples1, vector<vector<double>> &samples2, double T1, double T2, int i, double fend, double df, vector<complex<double>> &h2, vector<double> &noise, MatrixXd &fish1, MatrixXd &fish2, SelfAdjointEigenSolver<Eigen::MatrixXd> &es1 , SelfAdjointEigenSolver<Eigen::MatrixXd> &es2, int &counter){
    double likeT1X2 = loglike(loc2, 0, fend, df, h2, noise, T1);
    double likeT2X1 = loglike(loc1, 0, fend, df, h2, noise, T2);
    
//        cout << "Interchain swap proposed between T1 = " << T1 << " and T2 = " << T2 << endl;
//        cout << "T1X2 = " << likeT1X2 << endl;
//        cout << "T1X2 = " << likeT2X1 << endl;
//        cout << "Locations before " << endl;
//        cout_vec(loc1);
//        cout_vec(loc2);
    
    double rh = min(1., exp((likeT1X2 + likeT2X1)-(likehere1 + likehere2)));
    double urn = gsl_ran_flat(r, 0, 1.);
    
    if(rh >= urn){
 //             cout << "accepted" << endl;
        swap_loc(loc1, loc2);
        swap_fishers(fish1, fish2, es1, es2, T1, T2);
        likehere1 = likeT1X2;
        likehere2 = likeT2X1;
  //             cout << "Locations after " << endl;
        cout_vec(loc1);
        cout_vec(loc2);
//                cout << "Like 1 = " << likehere1 << " Like 2 = " << likehere2 << endl;
        write_vec_to_vec(samples1, loc1, i);
        write_vec_to_vec(samples2, loc2, i);
        counter++;
    } else {
 //             cout << "rejected" << endl;
        write_vec_to_vec(samples1, loc1, i);
        write_vec_to_vec(samples2, loc2, i);
    }
//       cout << endl;
}
void update_fisher(vector<double> &loc, MatrixXd &fish, SelfAdjointEigenSolver<Eigen::MatrixXd> &es, double T, double fend, double df, vector<double> &noise){
//    fish = fim(loc, noise, 0, fend, df, 1e-8, T, 0);
//    es.compute(fish);
//
    if(loc[2] > 0.1){
//        cout << "Location is " << endl;
//        cout << endl;
//        cout_vec(loc);
//        cout << endl;
        
        fish = fim(loc, noise, 0, fend, df, 1e-8, T, 1);
        es.compute(fish);
        
//        cout << "Fisher updated to " << endl;
//        cout << fish << endl;
//        cout << endl;
//        cout << "Eigen values are now " << endl;
//        cout << endl;
//        cout << es.eigenvalues() << endl;
//        cout << "Eigen Vectors are now " << endl;
//        cout << endl;
//        cout << es.eigenvectors() << endl;
//        cout << endl;
    }
    else {
//        cout << "Location is " << endl;
//        cout << endl;
//        cout_vec(loc);
//        cout << endl;
        
        vector<double> tmp_loc(5);
        set_loc(loc, tmp_loc);
        tmp_loc[2] = 0.1;
        fish = fim(tmp_loc, noise, 0, fend, df, 1e-8, T, 1);
        es.compute(fish);
        
//        cout << "Fisher updated to " << endl;
//        cout << fish << endl;
//        cout << endl;
//        cout << "Eigen values are now " << endl;
//        cout << endl;
//        cout << es.eigenvalues() << endl;
//        cout << "Eigen Vectors are now " << endl;
//        cout << endl;
//        cout << es.eigenvectors() << endl;
//        cout << endl;
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
int check_priors(vector<double> &prop){
    int cont = 0;
    if(prop[0] > 50 || prop[0] < 1) {cont = 1;}
    if(prop[1] > 0.25 || prop[1] < 0.15) {cont = 1;}
    if(prop[2] > 0.8 || prop[2] < 0.001) {cont = 1;}
    if(prop[3] > 150 || prop[3] < 40) {cont = 1;}
    if(prop[4] > 1e-16 || prop[4] < 5.7e-19) {cont = 1;}
    return cont;
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
