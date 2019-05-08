#include "fisher.hpp"

//compile with
// g++ -I/Users/blakemoore/eigen -I/Users/blakemoore/fftw/include -I/Users/blakemoore/gsl/include -O3 -ffast-math -g3 -c -fmessage-length=0 -std=c++14 -MMD -MP -MF"fisher.d" -MT"fisher.o" -o "fisher.o" "fisher.cpp"


double inner_product(vector<complex<double>> &h1, vector<complex<double>> &h2, vector<double> &noise, double df){
    double sum;
    int N = h1.size();
    sum = 0;
    for(int i = 0; i < N; i++){
        sum += real(4./exp(noise[i])*h1[i]*conj(h2[i]));
    }
    return sum*df;
}
vector<complex<double>> sum_f2(vector<vector<complex<double>>> &vect){
    int N = vect[0].size();
    vector<complex<double>> summed(N);
    int j = vect.size();
    for (int i = 0; i < N; i++){
        for(int k = 0; k < j; k++){
            summed[i] += vect[k][i];
        }
    }
    return summed;
}
vector<complex<double>> finite_diff(vector<complex<double>> &vect_right, vector<complex<double>> &vect_left, double ep){
    int N = vect_right.size();
    vector<complex<double>> deriv(N);
    for(int i = 0; i < N; i++){
        deriv[i] = (vect_right[i] - vect_left[i])/(2*ep);
    }
    return deriv;
}
vector<complex<double>> deriv_A(vector<complex<double>> &vect_right, double A){
    int N = vect_right.size();
    vector<complex<double>> deriv(N);
    for(int i = 0; i < N; i++){
        deriv[i] = (vect_right[i])/A;
    }
    return deriv;
}
vector<complex<double>> gen_waveform(double M, double eta, double e0, double p0, double A, double f0, double fend, double df){
    TaylorF2e F2e(M, eta, e0, p0, exp(A), f0, fend, df);
//    TaylorF2e F2e(M, eta, e0, p0, A, f0, fend, df);
    F2e.init_interps(1000);
    F2e.make_scheme();
    vector<vector<complex<double>>> vect;
    vect = F2e.get_F2e_min();
    vector<complex<double>> summedf2 = sum_f2(vect);
    return summedf2;
}
Eigen::MatrixXd fim(vector<double> &loc, vector<double> &noise, double f0, double fend, double df, double ep){
    //extract current location
    double M = loc[0];
    double eta = loc[1];
    double e0 = loc[2];
    double p0 = loc[3];
    double A = log(loc[4]);
//    double A = loc[4];
    
    //First generate the waveforms required to compute the numerical derivatives
    vector<complex<double>> M_right = gen_waveform(M + ep, eta, e0, p0, A, f0, fend, df);
    vector<complex<double>> M_left = gen_waveform(M - ep, eta, e0, p0, A, f0, fend, df);
    
    vector<complex<double>> eta_right = gen_waveform(M, eta + ep, e0, p0, A, f0, fend, df);
    vector<complex<double>> eta_left = gen_waveform(M, eta - ep, e0, p0, A, f0, fend, df);
    
    vector<complex<double>> e0_right = gen_waveform(M, eta, e0 + ep, p0, A, f0, fend, df);
    vector<complex<double>> e0_left = gen_waveform(M, eta, e0 - ep, p0, A, f0, fend, df);
    
    vector<complex<double>> p0_right = gen_waveform(M, eta, e0, p0 + ep, A, f0, fend, df);
    vector<complex<double>> p0_left = gen_waveform(M, eta, e0, p0 - ep, A, f0, fend, df);
    
    vector<complex<double>> A_right = gen_waveform(M, eta, e0, p0, A + ep, f0, fend, df);
    vector<complex<double>> A_left = gen_waveform(M, eta, e0, p0, A - ep, f0, fend, df);
//    vector<complex<double>> A_right = gen_waveform(M, eta, e0, p0, A, f0, fend, df);
    
    //Compute the numerical derivatives
    vector<complex<double>> M_deriv = finite_diff(M_right, M_left, ep);
    vector<complex<double>> eta_deriv = finite_diff(eta_right, eta_left, ep);
    vector<complex<double>> e0_deriv = finite_diff(e0_right, e0_left, ep);
    vector<complex<double>> p0_deriv = finite_diff(p0_right, p0_left, ep);
    vector<complex<double>> A_deriv = finite_diff(A_right, A_left, ep);
    
    //compute the elements of the fisher matrix;
    double prod_mm = inner_product(M_deriv, M_deriv, noise, df);
    double prod_meta = inner_product(M_deriv, eta_deriv, noise, df);
    double prod_me0 = inner_product(M_deriv, e0_deriv, noise, df);
    double prod_mp0 = inner_product(M_deriv, p0_deriv, noise, df);
    double prod_mA = inner_product(M_deriv, A_deriv, noise, df);
    double prod_etaeta = inner_product(eta_deriv, eta_deriv, noise, df);
    double prod_etae0 = inner_product(eta_deriv, e0_deriv, noise, df);
    double prod_etap0 = inner_product(eta_deriv, p0_deriv, noise, df);
    double prod_etaA = inner_product(eta_deriv, A_deriv, noise, df);
    double prod_e0e0 = inner_product(e0_deriv, e0_deriv, noise, df);
    double prod_e0p0 = inner_product(e0_deriv, p0_deriv, noise, df);
    double prod_e0A = inner_product(e0_deriv, A_deriv, noise, df);
    double prod_p0p0 = inner_product(p0_deriv, p0_deriv, noise, df);
    double prod_p0A = inner_product(p0_deriv, A_deriv, noise, df);
    double prod_AA = inner_product(A_deriv, A_deriv, noise, df);
    
    //Load up a fisher matrix
    Eigen::MatrixXd m(5,5);
    m(0,0) = prod_mm;
    m(1,1) = prod_etaeta;
    m(2,2) = prod_e0e0;
    m(3,3) = prod_p0p0;
    m(4,4) = prod_AA;
    
    m(0,1) = prod_meta;
    m(0,2) = prod_me0;
    m(0,3) = prod_mp0;
    m(0,4) = prod_mA;
    m(1,0) = prod_meta;
    m(2,0) = prod_me0;
    m(3,0) = prod_mp0;
    m(4,0) = prod_mA;
    
    m(1,2) = prod_etae0;
    m(1,3) = prod_etap0;
    m(1,4) = prod_etaA;
    m(2,1) = prod_etae0;
    m(3,1) = prod_etap0;
    m(4,1) = prod_etaA;
    
    m(2,3) = prod_e0p0;
    m(2,4) = prod_e0A;
    m(3,2) = prod_e0p0;
    m(4,2) = prod_e0A;
    
    m(3,4) = prod_p0A;
    m(4,3) = prod_p0A;
    
    return m;
}
Eigen::MatrixXd fim(vector<double> &loc, vector<double> &noise, double f0, double fend, double df, double ep, double T){
    return 1./T*fim(loc, noise, f0, fend, df, ep);
}
////////////////////////////////////////////////////////
//
// Here we're going to compute the fisher
// via Neil's suggestion
//
////////////////////////////////////////////////////////
vector<vector<complex<double>>> gen_waveform_full(double M, double eta, double e0, double p0, double A, double f0, double fend, double df){
    TaylorF2e F2e(exp(M), eta, e0, p0, exp(A), f0, fend, df);
    F2e.init_interps(1000);
    F2e.make_scheme();
    vector<vector<complex<double>>> vect;
    vect = F2e.get_F2e_min();
    return vect;
}
double prod_rev(vector<double> &Ai, vector<double> &Aj, vector<double> &A, vector<double> &Phii, vector<double> &Phij, vector<double> &noise, double &df){
    int N = Ai.size();
    double sum = 0;
    for(int i = 0; i < N; i++){
        sum += 4./exp(noise[i])*(Ai[i]*Aj[i] + A[i]*A[i]*Phii[i]*Phij[i]);
    }
    return sum*df;
}

double prod_rev(vector<vector<double>> &deriv_i, vector<vector<double>> &deriv_j, vector<vector<double>> &Amps, vector<double> &noise, double &df){
    int j = deriv_i.size();
    double sum = 0;
    for(int k = 0; k < j; k+=2){
        sum += prod_rev(deriv_i[k], deriv_j[k], Amps[k], deriv_i[k+1], deriv_j[k+1], noise, df);
//        cout << "sum = " << sum << "with j = " << k/2 << endl;
    }
    return sum;
}

vector<vector<double>> get_amp_phs(vector<vector<complex<double>>> &harm_wav){
    int N = harm_wav[0].size();
    int j = harm_wav.size();
    vector<vector<double>> amp_phs(2*j, vector<double> (N));
    complex<double> val = 0;
    for(int k = 0; k < j; k+=2){
        for(int i = 0; i < N; i++){
            val = harm_wav[k/2][i];
            amp_phs[k][i] = abs(val);
            amp_phs[k+1][i] = arg(val);
        }
    }
    return amp_phs;
}
vector<vector<double>> gen_amp_phs(double M, double eta, double e0, double p0, double A, double f0, double fend, double df){
    vector<vector<complex<double>>> harms = gen_waveform_full(M, eta, e0, p0, A, f0, fend, df);
    vector<vector<double>> amp_phs = get_amp_phs(harms);
    return amp_phs;
}

vector<double> finite_diff(vector<double> &vect_right, vector<double> &vect_left, double ep){
    int N = vect_right.size();
    vector<double> deriv(N);
    double diff = 0;
    for(int i = 0; i < N; i++){
        diff = vect_right[i] - vect_left[i];
        if((diff < 1) && (diff > -1)) {diff = diff;}
        else if (diff > 1) {diff = diff - 2*M_PI;}
        else if (diff < -1) {diff = diff + 2*M_PI;}
        deriv[i] = diff/(2*ep);
    }
    return deriv;
}

vector<vector<double>> finite_diff(vector<vector<double>> &vect_right, vector<vector<double>> &vect_left, double ep){
    int N = vect_right[0].size();
    int j1 = vect_right.size();
    int j2 = vect_left.size();
    int j = 0;
    if (j1 > j2) {j = j2;} else {j = j1;}
    vector<vector<double>> fds (j, vector<double> (N));
    for(int k = 0; k < j; k++){
        fds[k] = finite_diff(vect_right[k], vect_left[k], ep);
    }
    return fds;
}

Eigen::MatrixXd fim(vector<double> &loc, vector<double> &noise, double f0, double fend, double df, double ep ,int i){
    //extract current location
    double M = log(loc[0]);
    double eta = loc[1];
    double e0 = loc[2];
    double p0 = loc[3];
    double A = log(loc[4]);
    
    //First generate the waveforms required to compute the numerical derivatives (here ive just got the them as vectors with A_1, phi_1, .....)
    vector<vector<double>> M_right = gen_amp_phs(M + ep, eta, e0, p0, A, f0, fend, df);
    vector<vector<double>> M_left = gen_amp_phs(M - ep, eta, e0, p0, A, f0, fend, df);
    
    vector<vector<double>> eta_right = gen_amp_phs(M, eta + ep, e0, p0, A, f0, fend, df);
    vector<vector<double>> eta_left = gen_amp_phs(M, eta - ep, e0, p0, A, f0, fend, df);
    
    vector<vector<double>> e0_right = gen_amp_phs(M, eta, e0 + ep, p0, A, f0, fend, df);
    vector<vector<double>> e0_left = gen_amp_phs(M, eta, e0 - ep, p0, A, f0, fend, df);
    
    vector<vector<double>> p0_right = gen_amp_phs(M, eta, e0, p0 + ep, A, f0, fend, df);
    vector<vector<double>> p0_left = gen_amp_phs(M, eta, e0, p0 - ep, A, f0, fend, df);
    
    vector<vector<double>> A_right = gen_amp_phs(M, eta, e0, p0, A + ep, f0, fend, df);
    vector<vector<double>> A_left = gen_amp_phs(M, eta, e0, p0, A - ep, f0, fend, df);

    vector<vector<double>> A_gen = gen_amp_phs(M, eta, e0, p0, A, f0, fend, df);
    
    //Now the needed derivatives
    
    vector<vector<double>> M_deriv = finite_diff(M_right, M_left, ep);
    vector<vector<double>> eta_deriv = finite_diff(eta_right, eta_left, ep);
    vector<vector<double>> e0_deriv = finite_diff(e0_right, e0_left, ep);
    vector<vector<double>> p0_deriv = finite_diff(p0_right, p0_left, ep);
    vector<vector<double>> A_deriv = finite_diff(A_right, A_left, ep);
    
    //Now the inner products in the fisher
    double prod_mm = prod_rev(M_deriv, M_deriv, A_gen, noise, df);
    double prod_meta = prod_rev(M_deriv, eta_deriv, A_gen, noise, df);
    double prod_me0 = prod_rev(M_deriv, e0_deriv, A_gen, noise, df);
    double prod_mp0 = prod_rev(M_deriv, p0_deriv, A_gen, noise, df);
    double prod_mA = prod_rev(M_deriv, A_deriv, A_gen, noise, df);
    double prod_etaeta = prod_rev(eta_deriv, eta_deriv, A_gen, noise, df);
    double prod_etae0 = prod_rev(eta_deriv, e0_deriv, A_gen, noise, df);
    double prod_etap0 = prod_rev(eta_deriv, p0_deriv, A_gen, noise, df);
    double prod_etaA = prod_rev(eta_deriv, A_deriv, A_gen, noise, df);
    double prod_e0e0 = prod_rev(e0_deriv, e0_deriv, A_gen, noise, df);
    double prod_e0p0 = prod_rev(e0_deriv, p0_deriv, A_gen, noise, df);
    double prod_e0A = prod_rev(e0_deriv, A_deriv, A_gen, noise, df);
    double prod_p0p0 = prod_rev(p0_deriv, p0_deriv, A_gen, noise, df);
    double prod_p0A = prod_rev(p0_deriv, A_deriv, A_gen, noise, df);
    double prod_AA = prod_rev(A_deriv, A_deriv, A_gen, noise, df);
    
    //Load up a fisher matrix
    Eigen::MatrixXd m(5,5);
    m(0,0) = prod_mm + 1.38413;
    m(1,1) = prod_etaeta + 1200.;
    m(2,2) = prod_e0e0 + 18.75;
    m(3,3) = prod_p0p0 + 0.000991736;
    m(4,4) = prod_AA + 0.0017464;
    
    m(0,1) = prod_meta;
    m(0,2) = prod_me0;
    m(0,3) = prod_mp0;
    m(0,4) = prod_mA;
    m(1,0) = prod_meta;
    m(2,0) = prod_me0;
    m(3,0) = prod_mp0;
    m(4,0) = prod_mA;
    
    m(1,2) = prod_etae0;
    m(1,3) = prod_etap0;
    m(1,4) = prod_etaA;
    m(2,1) = prod_etae0;
    m(3,1) = prod_etap0;
    m(4,1) = prod_etaA;
    
    m(2,3) = prod_e0p0;
    m(2,4) = prod_e0A;
    m(3,2) = prod_e0p0;
    m(4,2) = prod_e0A;
    
    m(3,4) = prod_p0A;
    m(4,3) = prod_p0A;
    
    return m;
}

Eigen::MatrixXd fim_circ(vector<double> &loc, vector<double> &noise, double f0, double fend, double df, double ep ,int i){
    //extract current location
    double M = log(loc[0]);
    double eta = loc[1];
    double e0 = loc[2];
    double p0 = loc[3];
    double A = log(loc[4]);
    
    //First generate the waveforms required to compute the numerical derivatives (here ive just got the them as vectors with A_1, phi_1, .....)
    vector<vector<double>> M_right = gen_amp_phs(M + ep, eta, e0, p0, A, f0, fend, df);
    vector<vector<double>> M_left = gen_amp_phs(M - ep, eta, e0, p0, A, f0, fend, df);
    
    vector<vector<double>> eta_right = gen_amp_phs(M, eta + ep, e0, p0, A, f0, fend, df);
    vector<vector<double>> eta_left = gen_amp_phs(M, eta - ep, e0, p0, A, f0, fend, df);
    
//    vector<vector<double>> e0_right = gen_amp_phs(M, eta, e0 + ep, p0, A, f0, fend, df);
//    vector<vector<double>> e0_left = gen_amp_phs(M, eta, e0 - ep, p0, A, f0, fend, df);
    
//    vector<vector<double>> p0_right = gen_amp_phs(M, eta, e0, p0 + ep, A, f0, fend, df);
//    vector<vector<double>> p0_left = gen_amp_phs(M, eta, e0, p0 - ep, A, f0, fend, df);
    
    vector<vector<double>> A_right = gen_amp_phs(M, eta, e0, p0, A + ep, f0, fend, df);
    vector<vector<double>> A_left = gen_amp_phs(M, eta, e0, p0, A - ep, f0, fend, df);
    
    vector<vector<double>> A_gen = gen_amp_phs(M, eta, e0, p0, A, f0, fend, df);
    
    //Now the needed derivatives
    
    vector<vector<double>> M_deriv = finite_diff(M_right, M_left, ep);
    vector<vector<double>> eta_deriv = finite_diff(eta_right, eta_left, ep);
//    vector<vector<double>> e0_deriv = finite_diff(e0_right, e0_left, ep);
//    vector<vector<double>> p0_deriv = finite_diff(p0_right, p0_left, ep);
    vector<vector<double>> A_deriv = finite_diff(A_right, A_left, ep);
    
    //Now the inner products in the fisher
    double prod_mm = prod_rev(M_deriv, M_deriv, A_gen, noise, df);
    double prod_meta = prod_rev(M_deriv, eta_deriv, A_gen, noise, df);
//    double prod_me0 = prod_rev(M_deriv, e0_deriv, A_gen, noise, df);
//    double prod_mp0 = prod_rev(M_deriv, p0_deriv, A_gen, noise, df);
    double prod_mA = prod_rev(M_deriv, A_deriv, A_gen, noise, df);
    double prod_etaeta = prod_rev(eta_deriv, eta_deriv, A_gen, noise, df);
//    double prod_etae0 = prod_rev(eta_deriv, e0_deriv, A_gen, noise, df);
//    double prod_etap0 = prod_rev(eta_deriv, p0_deriv, A_gen, noise, df);
    double prod_etaA = prod_rev(eta_deriv, A_deriv, A_gen, noise, df);
//    double prod_e0e0 = prod_rev(e0_deriv, e0_deriv, A_gen, noise, df);
//    double prod_e0p0 = prod_rev(e0_deriv, p0_deriv, A_gen, noise, df);
//    double prod_e0A = prod_rev(e0_deriv, A_deriv, A_gen, noise, df);
//    double prod_p0p0 = prod_rev(p0_deriv, p0_deriv, A_gen, noise, df);
//    double prod_p0A = prod_rev(p0_deriv, A_deriv, A_gen, noise, df);
    double prod_AA = prod_rev(A_deriv, A_deriv, A_gen, noise, df);
    
    //Load up a fisher matrix
    Eigen::MatrixXd m(3,3);
    m(0,0) = prod_mm + 1.38413;
    m(1,1) = prod_etaeta + 1200.;
    m(2,2) = prod_AA + 0.0017464;
    
//    m(0,0) = prod_mm;
//    m(1,1) = prod_etaeta;
//    m(2,2) = prod_AA;

    
    m(0,1) = prod_meta;
//    m(0,2) = prod_me0;
//    m(0,3) = prod_mp0;
    m(0,2) = prod_mA;
    m(1,0) = prod_meta;
//    m(2,0) = prod_me0;
//    m(3,0) = prod_mp0;
    m(2,0) = prod_mA;
    
//    m(1,2) = prod_etae0;
//    m(1,3) = prod_etap0;
    m(1,2) = prod_etaA;
//    m(2,1) = prod_etae0;
//    m(3,1) = prod_etap0;
    m(2,1) = prod_etaA;
    
//    m(2,3) = prod_e0p0;
//    m(2,4) = prod_e0A;
//    m(3,2) = prod_e0p0;
//    m(4,2) = prod_e0A;
    
//    m(3,4) = prod_p0A;
//    m(4,3) = prod_p0A;
    return m;
}

Eigen::MatrixXd fim_ecc(vector<double> &loc, vector<double> &noise, double f0, double fend, double df, double ep ,int i){
    //extract current location
    double M = log(loc[0]);
    double eta = loc[1];
    double e0 = loc[2];
    double p0 = loc[3];
    double A = log(loc[4]);
    
    //First generate the waveforms required to compute the numerical derivatives (here ive just got the them as vectors with A_1, phi_1, .....)
    vector<vector<double>> M_right = gen_amp_phs(M + ep, eta, e0, p0, A, f0, fend, df);
    vector<vector<double>> M_left = gen_amp_phs(M - ep, eta, e0, p0, A, f0, fend, df);
    
    vector<vector<double>> eta_right = gen_amp_phs(M, eta + ep, e0, p0, A, f0, fend, df);
    vector<vector<double>> eta_left = gen_amp_phs(M, eta - ep, e0, p0, A, f0, fend, df);
    
    vector<vector<double>> e0_right = gen_amp_phs(M, eta, e0 + ep, p0, A, f0, fend, df);
    vector<vector<double>> e0_left = gen_amp_phs(M, eta, e0 - ep, p0, A, f0, fend, df);
    
    //    vector<vector<double>> p0_right = gen_amp_phs(M, eta, e0, p0 + ep, A, f0, fend, df);
    //    vector<vector<double>> p0_left = gen_amp_phs(M, eta, e0, p0 - ep, A, f0, fend, df);
    
    vector<vector<double>> A_right = gen_amp_phs(M, eta, e0, p0, A + ep, f0, fend, df);
    vector<vector<double>> A_left = gen_amp_phs(M, eta, e0, p0, A - ep, f0, fend, df);
    
    vector<vector<double>> A_gen = gen_amp_phs(M, eta, e0, p0, A, f0, fend, df);
    
    //Now the needed derivatives
    
    vector<vector<double>> M_deriv = finite_diff(M_right, M_left, ep);
    vector<vector<double>> eta_deriv = finite_diff(eta_right, eta_left, ep);
    vector<vector<double>> e0_deriv = finite_diff(e0_right, e0_left, ep);
    //    vector<vector<double>> p0_deriv = finite_diff(p0_right, p0_left, ep);
    vector<vector<double>> A_deriv = finite_diff(A_right, A_left, ep);
    
    //Now the inner products in the fisher
    double prod_mm = prod_rev(M_deriv, M_deriv, A_gen, noise, df);
    double prod_meta = prod_rev(M_deriv, eta_deriv, A_gen, noise, df);
    double prod_me0 = prod_rev(M_deriv, e0_deriv, A_gen, noise, df);
    //    double prod_mp0 = prod_rev(M_deriv, p0_deriv, A_gen, noise, df);
    double prod_mA = prod_rev(M_deriv, A_deriv, A_gen, noise, df);
    double prod_etaeta = prod_rev(eta_deriv, eta_deriv, A_gen, noise, df);
    double prod_etae0 = prod_rev(eta_deriv, e0_deriv, A_gen, noise, df);
    //    double prod_etap0 = prod_rev(eta_deriv, p0_deriv, A_gen, noise, df);
    double prod_etaA = prod_rev(eta_deriv, A_deriv, A_gen, noise, df);
    double prod_e0e0 = prod_rev(e0_deriv, e0_deriv, A_gen, noise, df);
    //    double prod_e0p0 = prod_rev(e0_deriv, p0_deriv, A_gen, noise, df);
    double prod_e0A = prod_rev(e0_deriv, A_deriv, A_gen, noise, df);
    //    double prod_p0p0 = prod_rev(p0_deriv, p0_deriv, A_gen, noise, df);
    //    double prod_p0A = prod_rev(p0_deriv, A_deriv, A_gen, noise, df);
    double prod_AA = prod_rev(A_deriv, A_deriv, A_gen, noise, df);
    
    //Load up a fisher matrix
    Eigen::MatrixXd m(4,4);
    m(0,0) = prod_mm + 1.38413;
    m(1,1) = prod_etaeta + 1200.;
    m(2,2) = prod_e0e0 + 18.75;
    m(3,3) = prod_AA + 0.0017464;
    
    m(0,1) = prod_meta;
    m(0,2) = prod_me0;
    //    m(0,3) = prod_mp0;
    m(0,3) = prod_mA;
    m(1,0) = prod_meta;
    m(2,0) = prod_me0;
    //    m(3,0) = prod_mp0;
    m(3,0) = prod_mA;
    
    m(1,2) = prod_etae0;
    //    m(1,3) = prod_etap0;
    m(1,3) = prod_etaA;
    m(2,1) = prod_etae0;
    //    m(3,1) = prod_etap0;
    m(3,1) = prod_etaA;
    
    //    m(2,3) = prod_e0p0;
    m(2,3) = prod_e0A;
    //    m(3,2) = prod_e0p0;
    m(3,2) = prod_e0A;
    
    //    m(3,4) = prod_p0A;
    //    m(4,3) = prod_p0A;
    return m;
}

Eigen::MatrixXd fim(vector<double> &loc, vector<double> &noise, double f0, double fend, double df, double ep , double T, int i){
    if(i == 1){
        Eigen::MatrixXd m = 1./T*fim_circ(loc, noise, f0, fend, df, ep, i);
        double condition_number;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::MatrixXd sings(3,3);
        sings(0,0) = svd.singularValues()(0);
        sings(1,1) = svd.singularValues()(1);
        sings(2,2) = svd.singularValues()(2);
        
        sings(0,1) = 0;
        sings(0,2) = 0;
        sings(1,0) = 0;
        sings(2,0) = 0;
        
        sings(1,2) = 0;
        sings(2,1) = 0;
        condition_number = sings(0,0)/sings(2,2);

        //    cout << svd.matrixU() << endl << endl;
        //    cout << sings << endl << endl;
        //    cout << svd.matrixV().adjoint() << endl << endl;
        //    cout << "make m  = " << svd.matrixU()*sings*svd.matrixV().adjoint() << endl << endl;
        //    cout << "actual m = " << m << endl << endl;
        while(condition_number > 5*1e5){
            sings(2,2) *= 1.1;
            m = svd.matrixU()*sings*svd.matrixV().adjoint();
            svd.compute(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
            sings(0,0) = svd.singularValues()(0);
            sings(1,1) = svd.singularValues()(1);
            sings(2,2) = svd.singularValues()(2);
            condition_number = sings(0,0)/sings(2,2);
        }
        //    cout << svd.matrixU() << endl << endl;
        //    cout << sings << endl << endl;
        //    cout << svd.matrixV().adjoint() << endl << endl;
        //    cout << "make m  = " << svd.matrixU()*sings*svd.matrixV().adjoint() << endl << endl;
        //    cout << "actual m = " << m << endl << endl;
        //    cout << "condition number = " << condition_number << endl;
        return m;
    } else if (i == 2){
        Eigen::MatrixXd m = 1./T*fim_ecc(loc, noise, f0, fend, df, ep, i);
        double condition_number;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::MatrixXd sings(4,4);
        sings(0,0) = svd.singularValues()(0);
        sings(1,1) = svd.singularValues()(1);
        sings(2,2) = svd.singularValues()(2);
        sings(3,3) = svd.singularValues()(3);
        
        sings(0,1) = 0;
        sings(0,2) = 0;
        sings(0,3) = 0;
        sings(1,0) = 0;
        sings(2,0) = 0;
        sings(3,0) = 0;
        
        sings(1,2) = 0;
        sings(1,3) = 0;
        sings(2,1) = 0;
        sings(3,1) = 0;
        
        sings(2,3) = 0;
        sings(3,2) = 0;
        condition_number = sings(0,0)/sings(3,3);
        
        //    cout << svd.matrixU() << endl << endl;
        //    cout << sings << endl << endl;
        //    cout << svd.matrixV().adjoint() << endl << endl;
        //    cout << "make m  = " << svd.matrixU()*sings*svd.matrixV().adjoint() << endl << endl;
        //    cout << "actual m = " << m << endl << endl;
        while(condition_number > 5*1e5){
            sings(3,3) *= 1.1;
            m = svd.matrixU()*sings*svd.matrixV().adjoint();
            svd.compute(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
            sings(0,0) = svd.singularValues()(0);
            sings(1,1) = svd.singularValues()(1);
            sings(2,2) = svd.singularValues()(2);
            sings(3,3) = svd.singularValues()(3);
            condition_number = sings(0,0)/sings(3,3);
        }
        //    cout << svd.matrixU() << endl << endl;
        //    cout << sings << endl << endl;
        //    cout << svd.matrixV().adjoint() << endl << endl;
        //    cout << "make m  = " << svd.matrixU()*sings*svd.matrixV().adjoint() << endl << endl;
        //    cout << "actual m = " << m << endl << endl;
        //    cout << "condition number = " << condition_number << endl;
        return m;
    } else {
        
        Eigen::MatrixXd m = 1./T*fim(loc, noise, f0, fend, df, ep, i);
        double condition_number;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::MatrixXd sings(5,5);
        sings(0,0) = svd.singularValues()(0);
        sings(1,1) = svd.singularValues()(1);
        sings(2,2) = svd.singularValues()(2);
        sings(3,3) = svd.singularValues()(3);
        sings(4,4) = svd.singularValues()(4);
        
        sings(0,1) = 0;
        sings(0,2) = 0;
        sings(0,3) = 0;
        sings(0,4) = 0;
        sings(1,0) = 0;
        sings(2,0) = 0;
        sings(3,0) = 0;
        sings(4,0) = 0;
        
        sings(1,2) = 0;
        sings(1,3) = 0;
        sings(1,4) = 0;
        sings(2,1) = 0;
        sings(3,1) = 0;
        sings(4,1) = 0;
        
        sings(2,3) = 0;
        sings(2,4) = 0;
        sings(3,2) = 0;
        sings(4,2) = 0;
        
        sings(3,4) = 0;
        sings(4,3) = 0;
        condition_number = sings(0,0)/sings(4,4);
        
        //    cout << svd.matrixU() << endl << endl;
        //    cout << sings << endl << endl;
        //    cout << svd.matrixV().adjoint() << endl << endl;
        //    cout << "make m  = " << svd.matrixU()*sings*svd.matrixV().adjoint() << endl << endl;
        //    cout << "actual m = " << m << endl << endl;
        while(condition_number > 5*1e5){
            sings(4,4) *= 1.1;
            m = svd.matrixU()*sings*svd.matrixV().adjoint();
            svd.compute(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
            sings(0,0) = svd.singularValues()(0);
            sings(1,1) = svd.singularValues()(1);
            sings(2,2) = svd.singularValues()(2);
            sings(3,3) = svd.singularValues()(3);
            sings(4,4) = svd.singularValues()(4);
            condition_number = sings(0,0)/sings(4,4);
        }
        //    cout << svd.matrixU() << endl << endl;
        //    cout << sings << endl << endl;
        //    cout << svd.matrixV().adjoint() << endl << endl;
        //    cout << "make m  = " << svd.matrixU()*sings*svd.matrixV().adjoint() << endl << endl;
        //    cout << "actual m = " << m << endl << endl;
        //    cout << "condition number = " << condition_number << endl;
        return m;
    }
}

void fisher_prop(vector<double> &loc, vector<double> &prop, Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es, const gsl_rng * r){
    double delt = gsl_ran_gaussian (r, 1.);
    int i = floor(gsl_ran_flat(r, 0, 5));
    if (i == 5){i = 4;}
    //    prop[0] = loc[0] + delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(0);
    prop[0] = loc[0]*exp(delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(0));
    prop[1] = loc[1] + delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(1);
    prop[2] = loc[2] + delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(2);
    prop[3] = loc[3] + delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(3);
    prop[4] = loc[4]*exp(delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(4));
}

void fisher_prop_circ(vector<double> &loc, vector<double> &prop, Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es, const gsl_rng * r){
    double delt = gsl_ran_gaussian (r, 1.);
    int i = floor(gsl_ran_flat(r, 0, 3));
    if (i == 3){i = 2;}
    //    prop[0] = loc[0] + delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(0);
    prop[0] = loc[0]*exp(delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(0));
    prop[1] = loc[1] + delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(1);
    prop[2] = loc[2];
    prop[3] = loc[3];
    prop[4] = loc[4]*exp(delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(2));
}

void fisher_prop_ecc(vector<double> &loc, vector<double> &prop, Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es, const gsl_rng * r){
    double delt = gsl_ran_gaussian (r, 1.);
    int i = floor(gsl_ran_flat(r, 0, 4));
    if (i == 4){i = 3;}
    //    prop[0] = loc[0] + delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(0);
    prop[0] = loc[0]*exp(delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(0));
    prop[1] = loc[1] + delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(1);
    prop[2] = loc[2] + delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(2);
    prop[3] = loc[3];
    prop[4] = loc[4]*exp(delt*1./sqrt((es.eigenvalues()(i)))*es.eigenvectors().col(i)(3));
}
