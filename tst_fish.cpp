#include <vector>
#include <iostream>
#include <math.h>
#include <fisher.hpp>


//g++ -I/Users/blakemoore/eigen -I/Users/blakemoore/fftw/include -I/Users/blakemoore/Desktop/Grad\ Research/MCMC_ecc -I/Users/blakemoore/gsl/include -O3 -ffast-math -g3 -c -fmessage-length=0 -std=c++14 -MMD -MP -MF"tst_fish.d" -MT"tst_fish.o" -o "tst_fish.o" "tst_fish.cpp"

// g++ -L/Users/blakemoore/gsl/lib -L/Users/blakemoore/fftw/lib -o "tst_fish"  ./Amps.o ./TaylorF2e.o ./fisher.o ./tst_fish.o  -lfftw3 -lgsl -lgslcblas -lm

using namespace std;
using namespace Eigen;

string path_inj = "/Users/blakemoore/Desktop/Grad Research/MCMC_ecc/injects/";
string path_noise = "/Users/blakemoore/Desktop/Grad Research/MCMC_ecc/noise/";
string path_write = "/Users/blakemoore/Desktop/Grad Research/MCMC_ecc/fisher_debug/";

vector<vector<double>> write_in_T4(string str, string path, double flim);
void write_vec_to_file(vector<complex<double>> &vect, string str, string path);
vector<vector<double>> write_in_F2(string str, string path, double flim);
void write_vec_to_file(vector<double> &vect, string str, string path);

int main(int argc, const char * argv[]){
    
    // write in data and create list of frequencies and a vector<complex> for the injection
    vector<vector<double>> data = write_in_F2("Inject_1.txt", path_inj, 1000);
    int data_size = data.size();
    vector<double> freqs(data_size);
    vector<complex<double>> h2(data_size);
    
    for(int i = 0; i < data_size; i++){
        freqs[i] = data[i][0];
        h2[i] = data[i][1] + 1i*data[i][2];
    }
    
    double df = freqs[1];
    double fend = freqs[data_size - 1];
    df = stod(argv[7]);
    int N = fend/df + 1;
    
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
    
//    vector<double> noise(data_size);
//
//    for (int i = 0; i < data_size; i++)    {
//        if (freqs[i] > 1 &&  freqs[i] < 4096 ) {
//            noise[i] = gsl_spline_eval (spline, (double) df*i, acc);
//        } else {
//            noise[i] = pow(10,10); //effectively make the noise infinite below 1Hz
//        }
//    }
    
    double f = 0;
    vector<double> noise(N);
    for (int i = 0; i < N; i++)    {
        f = df*i;
        if (f > 1 &&  f < 4096 ) {
            noise[i] = gsl_spline_eval (spline, f, acc);
        } else {
            noise[i] = pow(10,10); //effectively make the noise infinite below 1Hz
        }
    }
    
    double snrchk = inner_product(h2, h2, noise, df);
    cout << "Snr check = " << sqrt(snrchk) << endl;
    
    
    
    vector<double> loc(5);
    loc[0] = stod(argv[1]);
    loc[1] = stod(argv[2]);
    loc[2] = stod(argv[3]);
    loc[3] = stod(argv[4]);
    loc[4] = stod(argv[5]);
    
    double M = stod(argv[1]);
    double eta = stod(argv[2]);
    double e0 = stod(argv[3]);
    double p0 = stod(argv[4]);
    double A = stod(argv[5]);
    double ep = stod(argv[6]);
    
//    vector<vector<double>> A_gen = gen_amp_phs(M, eta, e0, p0, log(A), 0, fend, df);
//
//    vector<vector<double>> eta_right = gen_amp_phs(M, eta + ep, e0, p0, log(A), 0, fend, df);
//    vector<vector<double>> eta_left = gen_amp_phs(M, eta - ep, e0, p0, log(A), 0, fend, df);
//
//    vector<vector<double>> eta_deriv = finite_diff(eta_right, eta_left, ep);
//
//    double prod_etaeta = prod_rev(eta_deriv, eta_deriv, A_gen, noise, df);
//
//    cout << "eta eta comp fish = " << prod_etaeta << endl;
//
//    write_vec_to_file(eta_right[15] , "eta_right_15_"+to_string(stod(argv[7]))+".txt" , path_write);
//    write_vec_to_file(eta_left[15] , "eta_left_15_"+to_string(stod(argv[7]))+".txt", path_write);
//    write_vec_to_file(noise , "noise_"+to_string(stod(argv[7]))+".txt", path_write);
//    write_vec_to_file(freqs , "freq"+to_string(stod(argv[7]))+".txt", path_write);
    

//    MatrixXd fish = fim(loc, noise, 0, fend, df, ep);
//    SelfAdjointEigenSolver<Eigen::MatrixXd> es(5);
//    es.compute(fish);
//
//    cout << "fisher " << endl;
//    cout << fish << endl;
//    cout << "eigenvalues" << endl;
//    cout << es.eigenvalues() << endl;
//    cout << "eigenvectors" << endl;
//    cout << es.eigenvectors() << endl;

//    cout << "N of noise = " << N << endl;
    MatrixXd fish1 = fim(loc, noise, 0, fend, df, ep, 1.0, 1);
    SelfAdjointEigenSolver<Eigen::MatrixXd> es1(5);
    es1.compute(fish1);

    cout << "fisher (v2)" << endl;
    cout << fish1 << endl;
    cout << "eigenvalues (v2)" << endl;
    cout << es1.eigenvalues() << endl;
    cout << "eigenvectors (v2)" << endl;
    cout << es1.eigenvectors() << endl;

    //////Generate data
//    vector<complex<double>> model = gen_waveform(loc[0], loc[1], loc[2], loc[3], loc[4], 0, 4092, df);
//    write_vec_to_file(model, "Inject_"+to_string(stoi(argv[7]))+".txt", path_inj);
    
    return 0;
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
