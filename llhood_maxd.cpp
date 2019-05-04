#include "llhood_maxd.hpp"
using namespace std;

//compile with
// g++ -I/Users/blakemoore/fftw/include -I/Users/blakemoore/gsl/include -O3 -ffast-math -g3 -c -fmessage-length=0 -std=c++14 -MMD -MP -MF"llhood_maxd.d" -MT"llhood_maxd.o" -o "llhood_maxd.o" "llhood_maxd.cpp"

double get_snr_sq(vector<complex<double>> &vect, vector<double> &noise, double &df){
    double sum = 0;
    int N = vect.size();
    for(int i = 0; i < N; i++){
        sum += 4./exp(noise[i])*(vect[i].real()*vect[i].real() + vect[i].imag()*vect[i].imag());
    }
    return sum*df;
}

double get_snr_sq(vector<vector<complex<double>>> &vect, vector<double> &noise, double &df){
    double sum = 0;
    int j = vect.size();
    for(int i = 0; i < j; i++){
        sum += get_snr_sq(vect[i], noise, df);
    }
    return sum;
}

fftw_complex* take_invft_gj(vector<complex<double>> &h1, vector<complex<double>> &h2, vector<double> &noise){
    // prepare the inner product integrand
    fftw_complex *in, *out;
    int N = h2.size();
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    for(int i = 0; i < N; i++){
        in[i][0] = 4./exp(noise[i])*(h1[i].real()*h2[i].real() + h1[i].imag()*h2[i].imag()); //real part
        in[i][1] = 4./exp(noise[i])*(h1[i].real()*h2[i].imag() - h1[i].imag()*h2[i].real()); //imaginary part
    }
    
    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    
    fftw_free(in);
    return out;
}

vector<double> abs_gj(vector<complex<double>> &h1, vector<complex<double>> &h2, vector<double> &noise){
    fftw_complex *out;
    int N = h2.size();
    out = take_invft_gj(h1, h2, noise);
    vector<double> absvals(N);
    for (int i = 0; i < N; i++){
        absvals[i] = sqrt(out[i][0]*out[i][0] + out[i][1]*out[i][1]);
    }
    fftw_free(out);
    return absvals;
}


double inprod_h1_h2_maxd(vector<vector<complex<double>>> &h1, vector<complex<double>> &h2, vector<double> &noise, double &df){
    int size = h1.size();
    int lenth = h2.size();
    
    vector<vector<double>> hold_abs(size, vector<double>(lenth));
    for (int i = 0; i < size; i++){
        hold_abs[i] = abs_gj(h2, h1[i], noise);
    }
    vector<double> sum_abs(lenth);
    for (int i = 0; i < lenth; i++){
        for(int j = 0; j < size; j++){
            sum_abs[i] += hold_abs[j][i];
        }
    }
    double maxprod = *max_element(sum_abs.begin(), sum_abs.end());
    
    return maxprod*df;
}

double loglike(double M, double eta, double e0, double p0, double are, double aim, double f0, double fend, double df, vector<complex<double>> &h2, vector<double> &noise){
    TaylorF2e F2e(M, eta, e0, p0, are, aim, f0, fend, df);
    F2e.init_interps(1000);
    F2e.make_scheme();
    vector<vector<complex<double>>> vect;
    vect = F2e.get_F2e_min();
    
    double snrsq = get_snr_sq(vect, noise, df);
    double maxprod = inprod_h1_h2_maxd(vect, h2, noise, df);
    
    return (maxprod - snrsq*1./2.);
}

double loglike(double M, double eta, double e0, double p0, double ampmag, double f0, double fend, double df, vector<complex<double>> &h2, vector<double> &noise){
    TaylorF2e F2e(M, eta, e0, p0, ampmag, f0, fend, df);
    F2e.init_interps(1000);
    F2e.make_scheme();
    vector<vector<complex<double>>> vect;
    vect = F2e.get_F2e_min();
    
    double snrsq = get_snr_sq(vect, noise, df);
    double maxprod = inprod_h1_h2_maxd(vect, h2, noise, df);
    
    return (maxprod - snrsq*1./2.);
}

double loglike(vector<double> &params, double f0, double fend, double df, vector<complex<double>> &h2, vector<double> &noise){
    return loglike(params[0], params[1], params[2], params[3], params[4], f0, fend, df, h2, noise);
}
double loglike(vector<double> &params, double f0, double fend, double df, vector<complex<double>> &h2, vector<double> &noise, double T){
    return 1./T*loglike(params[0], params[1], params[2], params[3], params[4], f0, fend, df, h2, noise);
//    return 0;
}
