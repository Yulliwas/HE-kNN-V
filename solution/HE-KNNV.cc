#include <iostream>
#include <string.h>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>
#include "tfhe/tfhe.h"
#include <tfhe/tfhe_io.h>
#include "tfhe/polynomials.h"
#include "tfhe/tlwe.h"
#include "tfhe/tlwe_functions.h"
#include "tfhe/tfhe_garbage_collector.h"
#include "omp.h"
#define MSIZE 2000000
#define NBLINE 500
#define NBCOL 30
#define NBRESULT 1024
#define VERBOSE 1
#define TEST 1

using namespace std;
/**
 * @brief generation d'une clé TLWE
 * 
 * @param params 
 * @return TLweKey* 
 */
TLweKey *new_random_key(const TLweParams *params) {
        TLweKey *key = new_TLweKey(params);
        const int32_t N = params->N;
        const int32_t k = params->k;

        for (int32_t i = 0; i < k; ++i)
            for (int32_t j = 0; j < N; ++j)
                key->key[i].coefs[j] = rand() % 2;
        return key;
    }


/// Copier de Minelli
TFheGateBootstrappingParameterSet *our_default_gate_bootstrapping_parameters(int minimum_lambda)
{
    if (minimum_lambda > 128)
        cerr << "Sorry, for now, the parameters are only implemented for about 128bit of security!\n";

    static const int n = 1024;
    static const int N = 1024;
    static const int k = 1;
    static const double max_stdev = 1./(double)pow(10,9);

    static const int bk_Bgbit    = 64;  
    static const int bk_l        = 6;
    static const double bk_stdev = max_stdev; 

    static const int ks_basebit  = 1; 
    static const int ks_length   = 18;
    static const double ks_stdev = max_stdev;


    LweParams  *params_in    = new_LweParams (n,    ks_stdev, max_stdev);
    TLweParams *params_accum = new_TLweParams(N, k, bk_stdev, max_stdev);
    TGswParams *params_bk    = new_TGswParams(bk_l, bk_Bgbit, params_accum);

    TfheGarbageCollector::register_param(params_in);
    TfheGarbageCollector::register_param(params_accum);
    TfheGarbageCollector::register_param(params_bk);

    return new TFheGateBootstrappingParameterSet(ks_length, ks_basebit, params_in, params_bk);
}
/**
 * @brief construction de Int Polynomial (M dans le papier)
 * 
 * @param vec 
 * @param N 
 * @return IntPolynomial* 
 */
IntPolynomial* vectorSrcToIntPolynomial(float vec[NBCOL],int N,int nb_col,int v,int tau){
    IntPolynomial* t=new_IntPolynomial(N);
   // if (VERBOSE) cout << "vectorSrcToIntPolynomial"<<endl;
    for(int i=0;i<nb_col;i++){
        t->coefs[i]=round(vec[29-i]*tau/v);    
           
    }
    for(int i=nb_col;i<N;i++){
        t->coefs[i]=0;        
    }
    return t;
}

/**
 * @brief construction de torus polynomial pour les vecteurs de model (Ci dans le papier)
 * 
 * @param vec 
 * @param N 
 * @return TorusPolynomial* 
 */
TorusPolynomial* vectorMdlToTorusPolynomial(float vec[NBCOL],int N,int nb_col,int v,int tau){
    TorusPolynomial* t=new_TorusPolynomial(N);
    //if (VERBOSE) cout << "vectorMdlToTorusPolynomial"<<endl;
    for(int i=0;i<nb_col;i++){
        
       t->coefsT[i]=dtot32(vec[i]/(double)(v*tau));    
    }
    for(int i=nb_col;i<N;i++){
        t->coefsT[i]=0;        
    }
    return t;
}

/**
 * @brief Calcul de la formule de Ai 
 * 
 * @param vec 
 * @param N 
 * @param v 
 * @return TorusPolynomial* 
 */

TorusPolynomial* calculerAi(float vec[NBCOL],int32_t N,int v,int nb_col){
    TorusPolynomial* t=new_TorusPolynomial(N);
    double sum =0;
    for(int l=0;l<nb_col;l++){
        sum+=pow(vec[l]/(float)v,2);
    }
    //if (VERBOSE) cout << "somme Ai "<< sum <<endl;
    t->coefsT[nb_col-1]=dtot32(sum);
    return t;
}

/**
 * @brief Multiplier IntPolynomial avec un tlweSample
 * 
 * @param result 
 * @param tlwe 
 * @param testvect 
 * @param rparams 
 */
void multiplyPolynomial(TLweSample *result, const TLweSample *tlwe, const IntPolynomial *testvect, const TLweParams *rparams) {
       
    // results = (a,b)*testvect
    tLweClear(result,rparams);
    torusPolynomialMultNaive(result->a, testvect, tlwe->a);
    torusPolynomialMultNaive(result->b, testvect, tlwe->b);
    
}




/**
 * @brief Lecture de dataset 
 * 
 * @param dataset 
 * @param index 
 */
void lireDataSet(float dataset[NBLINE][NBCOL], int classes[NBLINE],int d,int nb_col){
    FILE* data = NULL; FILE* label = NULL;
    data = fopen("data.data","r+");
    label = fopen("label.data","r+");
    if (data != NULL && label != NULL)
    {
        for(int i=0;i<d;i++){
            fscanf(label,"%d",&classes[i]);
            for(int j=0;j<nb_col;j++){
                fscanf(data,"%f",&dataset[i][j]);
            }
        }
        fclose(data); 
        fclose(label); 
    }
    else
    {
        printf("Impossible d'ouvrir le fichier data.data ou label.data");
    }
    if (VERBOSE) cout << "Verification lecture dataset"<<dataset[0][0]<<endl;

}

/**
 * @brief Encoder le dataset en un veteur de torusPolynomial
 * 
 * @param dataset 
 * @param datasetEncoded 
 * @param N 
 */
void encodeDataset(float dataset[NBLINE][NBCOL], TorusPolynomial *datasetEncoded[NBLINE],int N,int d,int nb_col,int v,int tau){
    for(int i=0;i<d;i++){
        datasetEncoded[i]=vectorMdlToTorusPolynomial(dataset[i],N,nb_col,v,tau);
      // if (VERBOSE) cout <<" test encodage"<< t32tod(datasetEncoded[i]->coefsT[0])*300 << "   vrai "<< dataset[i][0] <<endl;
    }
}

/**
 * @brief Calculer tout les Ai et les encrypter (voir papier Zuber)
 * 
 * @param dataset 
 * @param Ai_vect 
 * @param N 
 * @param v 
 * @param alpha 
 * @param key1024_1 
 */
void calculerAi_vect(float dataset[NBLINE][NBCOL],TLweSample *Ai_vect[NBLINE], int N, int v,double alpha,  const  TLweKey *key1024_1,int d,int nb_col){
    //calculer les Ai en clair
    TorusPolynomial *Ai[d];
    for(int i=0;i<d; i++){
        Ai[i]=calculerAi(dataset[i],N,v,nb_col);
    }
    //encrypt Ai dans Ai_vect
    for(int i=0;i<d; i++){
        tLweSymEncrypt(Ai_vect[i], Ai[i], alpha, key1024_1);
    }
}

void calculerAi_vectClair(float dataset[NBLINE][NBCOL],TorusPolynomial *Ai_vect[NBLINE], int N, int v,int d,int nb_col){
    //calculer les Ai en clair
    
    for(int i=0;i<d; i++){
        Ai_vect[i]=calculerAi(dataset[i],N,v,nb_col);
    }
    
}

/**
 * @brief Encrypter le dataset encodée en utilisant tlwe
 * 
 * @param datasetEncoded 
 * @param datasetEncrypted 
 * @param alpha 
 * @param key1024_1 
 */
void encrypt_dataset(TorusPolynomial *datasetEncoded[NBLINE] ,TLweSample *datasetEncrypted[NBLINE],double alpha,  const TLweKey *key1024_1,int d){
        //encrypt Ai dans Ai_vect
    for(int i=0;i<d; i++){
        datasetEncrypted[i]=new_TLweSample(key1024_1->params);
        tLweSymEncrypt(datasetEncrypted[i], datasetEncoded[i], alpha, key1024_1);
        
    }
}
void tfhe_MuxRotate_FFT(TLweSample *result, const TLweSample *accum, const TGswSampleFFT *bki, const int32_t barai,
                        const TGswParams *bk_params) {
    // ACC = BKi*[(X^barai-1)*ACC]+ACC             //==[h+(X^barai-1).BKi]*ACC
    // temp = (X^barai-1)*ACC
    tLweMulByXaiMinusOne(result, barai, accum, bk_params->tlwe_params);
        
    // temp *= BKi
    tGswFFTExternMulToTLwe(result, bki, bk_params);
    // ACC += temp
    tLweAddTo(result, accum, bk_params->tlwe_params);
}




void tfhe_blindRotate_FFT_modified(TLweSample *accum,
                                 const TGswSampleFFT *bkFFT,
                                 const int32_t *bara,
                                 const int32_t n,
                                 const TGswParams *bk_params) {

    //TGswSampleFFT* temp = new_TGswSampleFFT(bk_params);
    TLweSample *temp = new_TLweSample(bk_params->tlwe_params);
    TLweSample *temp2 = temp;
    TLweSample *temp3 = accum;

    for (int32_t i = 0; i < n; i++) {
        const int32_t barai = bara[i];
        if (barai == 0) continue; //indeed, this is an easy case!
        //clock_t begin=clock();
        tfhe_MuxRotate_FFT(temp2, temp3, bkFFT + i, barai, bk_params);
        //cout << "temps blindrotate "<< clock()-begin<<endl;
        swap(temp2, temp3);
    }
    if (temp3 != accum) {
        tLweCopy(accum, temp3, bk_params->tlwe_params);
    }
    delete_TLweSample(temp);
    //delete_TGswSampleFFT(temp);
}   
    
void tfhe_blindRotateAndExtract_FFT_modified(LweSample *result[NBLINE],
                                           const TorusPolynomial *v,
                                           const TGswSampleFFT *bk,
                                           const int32_t barb,
                                           const int32_t *bara,
                                           const int32_t n,
                                           const TGswParams *bk_params,
                                           int16_t k,int16_t m) {

    const TLweParams *accum_params = bk_params->tlwe_params;
    const LweParams *extract_params = &accum_params->extracted_lweparams;
    const int32_t N = accum_params->N;
    const int32_t _2N = 2 * N;

    // Test polynomial 
    TorusPolynomial *testvectbis = new_TorusPolynomial(N);
    
    // Accumulator
    TLweSample *acc = new_TLweSample(accum_params);




    // testvector = X^{2N-barb}*v
   if (barb != 0) torusPolynomialMulByXai(testvectbis,_2N- barb, v);
   else torusPolynomialCopy(testvectbis, v);
    //torusPolynomialMulByXai(testvectbis, N-barb, v);
    tLweNoiselessTrivial(acc, testvectbis, accum_params);
    // Blind rotation
     
    
    
    
    tfhe_blindRotate_FFT(acc, bk, bara, n, bk_params);
    
    // Extraction
    int d;
    for(int16_t i=0; i<k;i++){
        d=round((double)i*4*N/(double)(4*m-4/*-4*/));
        tLweExtractLweSampleIndex(result[i], acc, d, extract_params, accum_params);
    }
        
    delete_TLweSample(acc);
    delete_TorusPolynomial(testvectbis);
}




/**
 * result = LWE(mu) iff phase(x)>0, LWE(-mu) iff phase(x)<0
 * @param result The resulting LweSample
 * @param bk The bootstrapping + keyswitch key
 * @param mu The output message (if phase(x)>0)
 * @param x The input sample
 */
void tfhe_bootstrap_woKS_FFT_modified(LweSample *result[NBLINE],
                                    const LweBootstrappingKeyFFT *bk,
                                    Torus32 mu,
                                    const LweSample *x,
                                    int16_t k,int16_t m) {
    
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t Nx2 = 2 * N;
    const int32_t n = in_params->n;
   

    TorusPolynomial *testvect = new_TorusPolynomial(N);
    int32_t *bara = new int32_t[N];

    
    // Modulus switching
    int32_t barb = modSwitchFromTorus32(x->b, Nx2);
    
    for (int32_t i = 0; i < n; i++) {
        bara[i] = modSwitchFromTorus32(x->a[i], Nx2);
    }
   
    // the initial testvec = [mu,mu,mu,...,mu]
    for (int32_t i = 0; i < N; i++) testvect->coefsT[i] = mu;
    
        
    // Bootstrapping rotation and extraction
    tfhe_blindRotateAndExtract_FFT_modified(result, testvect, bk->bkFFT, barb, bara, n, bk_params,k,m);
    

    delete[] bara;
    delete_TorusPolynomial(testvect);
}




/**
 * result = LWE(mu) iff phase(x)>0, LWE(-mu) iff phase(x)<0
 * @param result The resulting LweSample
 * @param bk The bootstrapping + keyswitch key
 * @param mu The output message (if phase(x)>0)
 * @param x The input sample
 */
void homomorphicSign(LweSample *result[NBLINE],
                               const LweBootstrappingKeyFFT *bk,
                               Torus32 mu,
                               const LweSample *x,
                               int16_t k,int16_t m) {

    LweSample *u[NBRESULT];
    for(int i=0;i<k;i++)
        u[i]= new_LweSample(&bk->accum_params->extracted_lweparams);
   
  // TorusPolynomial *testvect = new_TorusPolynomial(N);

   tfhe_bootstrap_woKS_FFT_modified(u, bk, mu, x,k,m);
   
  //  tfhe_bootstrap_woKS_FFT(u[0], bk, mu, x);
    // Key switching
    for(int i=1;i<k+1;i++)
       // lweKeySwitch(result[i], bk->ks, u[i]);
       lweCopy(result[i-1],u[i-1],&bk->accum_params->extracted_lweparams);
    
    for(int i=0;i<k;i++)
        delete_LweSample(u[i]);
}


/**
 * @brief formule 2 de papier zuber  (marche tres bien)
 * 
 * @param result 
 * @param Ai 
 * @param Aj 
 * @param Ci 
 * @param Cj 
 * @param M 
 * @param params 
 */
void formule2(LweSample *result, float Ci[NBCOL], float Cj[NBCOL],float Ai_vector[NBCOL], LweSample *MEncrypted[NBCOL],const LweParams *params,int nb_col){
    //Ai-Aj Ai is the norm of the vector Ci
    float Ai=0.; 
    float Aj=0.; 
    for(int i=0;i<nb_col;i++){
        Ai+=pow(Ci[i],2);
        Aj+=pow(Cj[i],2);
    }
    //Ai=sqrt(Ai);Aj=sqrt(Aj);
    Ai=(Ai-Aj)/16;

    float Cj_Ci[NBCOL];
    for(int i=0;i<nb_col;i++){
        Cj_Ci[i]=round(100.*(Cj[i]-Ci[i])/4.);
    }
   
   
    LweSample *somme=new_LweSample(params); 
    lweClear(somme,params);
    for (int l=0;l<nb_col;l++){
        lweAddMulTo(somme,Cj_Ci[l],MEncrypted[l],params);
    }
    LweSample *Ai_Aj=new_LweSample(params);
    lweNoiselessTrivial(Ai_Aj,dtot32(Ai),params);
    lweAddMulTo(Ai_Aj,2,somme,params);
    lweCopy(result,Ai_Aj,params);
}

float formule2Clair(float Ci[NBCOL], float Cj[NBCOL], float M[NBCOL],int nb_col){
    int params=1024;
    float Ai=0.; 
    float Aj=0.; 
    for(int i=0;i<nb_col;i++){
        Ai+=pow(Ci[i],2);
        Aj+=pow(Cj[i],2);
    }
    //Ai=sqrtAi);Aj=sqrt(Aj);
    Ai=(Ai-Aj)/16;

    int Cj_Ci[NBCOL];
    for(int i=0;i<nb_col;i++){
        Cj_Ci[i]=round(100.*(Cj[i]-Ci[i])/4.);
    }
    float somme=0.; 
    for (int l=0;l<nb_col;l++){
        somme+=M[l]/(400.)*Cj_Ci[l];
    }
    return Ai+somme*2;
}
/**
 * @brief Calculer une seule valeur de delta  (Verifier la fonction signe)
 * 
 * @param deltaValue 
 * @param Ai 
 * @param Aj 
 * @param Ci 
 * @param Cj 
 * @param M 
 * @param params 
 */
void deltaValueUnique(LweSample *deltaValue, float Ci[NBCOL], float Cj[NBCOL], LweSample *MEncrypted[NBCOL],const LweParams *params,const LweBootstrappingKeyFFT *bs_key,int m,int nb_col){
    LweSample *formula2 = new_LweSample(params);
    lweClear(formula2,params);
    clock_t  begin=clock();
    float *Ai_vector;
    formule2(formula2, Ci,Cj, Ai_vector,MEncrypted,params,nb_col);
    clock_t  end=clock();
   // cout << "Temps formule " <<  (end-begin) << "secondes" << endl <<endl;
     
    //do a sign bootstrapping 
    int32_t b_delta=(4*m/*-4*/);//4m-4 parameter dans Zuber
    Torus32 MU = modSwitchToTorus32(1,b_delta);
    
    tfhe_bootstrap_woKS_FFT(deltaValue,bs_key,MU,formula2);
   // lweCopy(deltaValue,formula2,&params->extracted_lweparams);
    //convert to 0 or 1
    LweSample *toAdd=new_LweSample(params);
    lweNoiselessTrivial(toAdd,  MU, params);
    
    lweAddTo(deltaValue,toAdd,params);
    
}


void deltaValueUnique2(LweSample *deltaValue, float Ci[NBCOL], float Cj[NBCOL], LweSample *MEncrypted[NBCOL],const LweParams *params,const LweBootstrappingKeyFFT *bs_key,int m,int nb_col){
    LweSample *formula2 = new_LweSample(params);
    lweClear(formula2,params);
    clock_t  begin=clock();
    float *Ai_vector;
    formule2(formula2, Ci,Cj, Ai_vector,MEncrypted,params,nb_col);
    clock_t  end=clock();
   // cout << "Temps formule " <<  (end-begin) << "secondes" << endl <<endl;
     
    //do a sign bootstrapping 
    //int32_t b_delta=(4*m/*-4*/);//4m-4 parameter dans Zuber
    //Torus32 MU = modSwitchToTorus32(1,b_delta);
    
    //tfhe_bootstrap_woKS_FFT(deltaValue,bs_key,MU,formula2);
    lweCopy(deltaValue,formula2,params);
    //convert to 0 or 1
    //LweSample *toAdd=new_LweSample(params);
    //lweNoiselessTrivial(toAdd,  MU, params);
    
    //lweAddTo(deltaValue,toAdd,params);
    
}

/**
 * @brief Construire la matrice deltaValue (Marche tres bien)
 * 
 * @param datasetEncoded 
 * @param datasetEncrypted 
 * @param alpha 
 * @param key1024_1 
 */
void deltaValuesMatrix(LweSample *deltaValues[NBLINE][NBLINE],float dataset[NBLINE][NBCOL],LweSample *MEncrypted[NBCOL],  const LweParams *params,const LweBootstrappingKeyFFT *bs_key,int m,int d,int nb_col){
    const LweParams *in_out_params = bs_key->in_out_params;
    static const Torus32 MU = modSwitchToTorus32(1,2*m);
/*
    //calcul premiere ligne 
    lweClear(deltaValues[0][0],params);
    #pragma omp parallel for
    for(int j=1;j<d;j++){
        deltaValueUnique2(deltaValues[0][j],dataset[0],dataset[j],MEncrypted,params,bs_key,m,nb_col);
    }

    //calcul des autres lignes
    #pragma omp parallel for schedule(dynamic)
    for(int i=1;i<d;i++){
        lweClear(deltaValues[i][i],params);
        for(int j=i+1;j<d;j++){
            lweCopy(deltaValues[i][j],deltaValues[0][j],params);
            lweSubTo(deltaValues[i][j],deltaValues[0][i],params);
        }
    }

     int32_t b_delta=(4*m);//4m-4 parameter dans Zuber
    Torus32 MU2 = modSwitchToTorus32(1,b_delta);
    //bootstrap en parallele
    #pragma omp parallel for schedule(dynamic)
    for(int i=0;i<d;i++){
        lweClear(deltaValues[i][i],params);
        for(int j=i+1;j<d;j++){
            tfhe_bootstrap_woKS_FFT(deltaValues[i][j],bs_key,MU2,deltaValues[i][j]);
            // lweCopy(deltaValue,result,&params->extracted_lweparams);
            //convert to 0 or 1
            LweSample *toAdd=new_LweSample(params);
            lweNoiselessTrivial(toAdd,  MU2, params);
    
            lweAddTo(deltaValues[i][j],toAdd,params);

            //dji=1-dij
            lweNoiselessTrivial(deltaValues[j][i],  MU2, in_out_params);
            lweSubTo(deltaValues[j][i],deltaValues[i][j],params);
        }
    }
*/
   
    
    

   // #pragma omp parallel for schedule(dynamic)
    for(int i=0;i<d;i++){
        lweClear(deltaValues[i][i],params);
        
        for(int j=i+1;j<d;j++){
            
            //cout << " i j "<<i << j <<endl;
            deltaValueUnique(deltaValues[i][j],dataset[i],dataset[j],MEncrypted,params,bs_key,m,nb_col);
            //dji=1-dij
            lweNoiselessTrivial(deltaValues[j][i],  MU, in_out_params);
            lweSubTo(deltaValues[j][i],deltaValues[i][j],params);
        }
        
    }
   
}

void scoreBlock(LweSample *deltaVectorValue,LweSample *sum,const LweBootstrappingKeyFFT *bk,const LweParams *params,int m,int16_t k){
    LweSample *result[NBRESULT];
    LweSample *sumCopy=new_LweSample(params);
    
    for(int i=0;i<NBRESULT;i++){
        result[i]=new_LweSample(params);
    }
    homomorphicSign(result,bk,modSwitchToTorus32(1,/*2**/(4*m-4)),sum,k,m);
    //pour avoir les valeur 0 et 1
    LweSample *toAdd=new_LweSample(&bk->accum_params->extracted_lweparams);
    static const Torus32 MU2 = modSwitchToTorus32(1,/*2**/(m*4-4));
    lweNoiselessTrivial(toAdd,  MU2, &bk->accum_params->extracted_lweparams);
    for(int c=0;c<k;c++){    
        lweNegate(result[c], result[c], &bk->accum_params->extracted_lweparams);
        lweAddTo(result[c],toAdd,&bk->accum_params->extracted_lweparams);     
    }
    //somme
    lweClear(sumCopy,params);
    for(int16_t j=0;j<k;j++)
        lweAddTo(sumCopy,result[j],params);
    lweCopy(deltaVectorValue,sumCopy,params);
 
}

//Implement and operation 

void scoreOperation(LweSample *deltaVector[NBLINE],LweSample *deltaValues[NBLINE][NBLINE],const LweBootstrappingKeyFFT *bk,const LweParams *params,int d,int m,int16_t k){
    //variables a utiliser 
    LweSample *result[NBRESULT];
    LweSample *sum=new_LweSample(params);
    int compteBlock;
    int cpt;
    int compteurAddition;
    bool premierBloc=true;

    LweSample *toAdd=new_LweSample(&bk->accum_params->extracted_lweparams);
    Torus32 MU2; 
    int m_size=2*m-2;
    
    for(int i=0;i<NBRESULT;i++){
        result[i]=new_LweSample(params);
    }

        
    
    
    
    //partie à faire 
    for(int i=0;i<d;i++){
        compteurAddition=0;
        premierBloc=true;
        lweClear(deltaVector[i],params);
        compteBlock=0;
        int room=m-k;
        
        for(int j=0;j<d;j++){
            
            if (j==i){
                continue;
            } 

            compteurAddition++;
            //cout << "ici "<< i<<j <<"   "<<compteurAddition<<endl;
            //effectuer les additions
            
            lweAddTo(deltaVector[i],deltaValues[j][i],params);
            
            

            

            if(compteurAddition==(m) && premierBloc==true){
                //construire le premier bloc
                premierBloc=false;
                compteurAddition=0;
                //cout << "  j+1=m"<< i<<" "<< j<< endl;
                scoreBlock(deltaVector[i],deltaVector[i],bk,params,m,k);
                //cout << "Fin de premier block " << i<< endl;
            }
            //cout << " j avant 2eme blc "<<j <<endl;
            else if ( premierBloc==false){
                room--;
               // cout << "room " << room <<endl;
                //A partir de 2eme block ERROR  
                if ((compteurAddition)%(m-k)==0){//c pas ca? 
               // cout << "  j+1%(m-k)==0  "<< i <<j<< endl;
                 //   cout << "  room"<<room<< endl;
                    compteBlock++;
                    room=m-k;
                    scoreBlock(deltaVector[i],deltaVector[i],bk,params,m,k);
                   // cout << "Fin de block " << compteBlock<< endl;
                }
                
            }
            
        }
       // cout << "room     vrai " << room <<endl;
        
        if (d>m){
            if(room==m-k){
                MU2=modSwitchToTorus32(room-1,m_size);
                lweNoiselessTrivial(toAdd,  MU2, &bk->accum_params->extracted_lweparams);
                lweAddTo(deltaVector[i],toAdd,params);
            } else{
                MU2 = modSwitchToTorus32(room,m_size);
                lweNoiselessTrivial(toAdd,  MU2, &bk->accum_params->extracted_lweparams);
                lweAddTo(deltaVector[i],toAdd,params);
            }
        }else if (d==m){
            
            MU2 = modSwitchToTorus32(1,m_size);
            lweNoiselessTrivial(toAdd,  MU2, &bk->accum_params->extracted_lweparams);
            lweAddTo(deltaVector[i],toAdd,params);
        } else{
            
            MU2 = modSwitchToTorus32(m-d+1,m_size);
            lweNoiselessTrivial(toAdd,  MU2, &bk->accum_params->extracted_lweparams);
            lweAddTo(deltaVector[i],toAdd,params);
        }
        

    }
    


    //somme apres max padding 
    for(int i=0;i<d;i++){
    // cout << "derniere somme "<<endl;
       LweSample *toAdd=new_LweSample(&bk->accum_params->extracted_lweparams);
      /*  Torus32 MU2 = modSwitchToTorus32(maxpadding,2*m-2);
        lweNoiselessTrivial(toAdd,  MU2, &bk->accum_params->extracted_lweparams);
        lweAddTo(deltaVector[i],toAdd,params);*/
       homomorphicSign(result,bk,modSwitchToTorus32(1,4*m-4),deltaVector[i],k,m);
      // MU2 = modSwitchToTorus32(1,4);
        lweNoiselessTrivial(toAdd,  modSwitchToTorus32(1,4*m-4), &bk->accum_params->extracted_lweparams);
        for(int j=0;j<k;j++){
        
            lweNegate(result[j], result[j], &bk->accum_params->extracted_lweparams);
            
            //lweAddTo(result[j],toAdd,&bk->accum_params->extracted_lweparams);
            
        }
        lweCopy(deltaVector[i],result[k-1],params); 
    
    
}
}

bool approxEquals(float a, float b) { return abs(a - b) < 0.01; }
bool signEgal(double a,double b){
    if ((a>=0 && b>=0) || (a<0 && b<0)) return true;
    return false;
}





void KNN(LweSample *deltaVector[NBLINE],LweSample *classVector[NBLINE],float dataset[NBLINE][NBCOL],int labels[NBLINE],LweSample *M[NBCOL],const LweParams *params, TFheGateBootstrappingCloudKeySet *bootk,int d,int m,int k,int nb_col){
    /**************************************************************************************/
    /************************* Constrction de delta matrices ******************************/
    if (VERBOSE) cout << "Construction la matrice des delta values"<<endl;
    
    LweSample *deltaValues[NBLINE][NBLINE];
    for(int i=0;i<d;i++){
        for(int j=0;j<d;j++){
            deltaValues[i][j]=new_LweSample(params);
        }
    }
    clock_t begin = clock();
    deltaValuesMatrix(deltaValues,dataset,M, params,bootk->bkFFT,m,d,nb_col);
   
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC; 
    if (VERBOSE && TEST) cout << "Temps construction de deltamatrices" <<  time_spent<< endl;
    
    
    /**************************************************************************************/
    /******************************* Scoring operation ************************************/
    
    cout <<endl <<"Scoring operation   "<< endl ;
    
    begin = clock();

    scoreOperation(deltaVector,deltaValues,bootk->bkFFT,bootk->bkFFT->extract_params,d,m,k);

    LweSample *label=new_LweSample(bootk->bkFFT->in_out_params);
    LweSample *temp=new_LweSample(bootk->bkFFT->in_out_params);
    LweSample *temp2=new_LweSample(bootk->bkFFT->in_out_params);
    //Appliquer AND operation with between deltavector and datasetlabels
    LweSample *temp_result = new_LweSample(bootk->bkFFT->in_out_params);
    static const Torus32 AndConst = modSwitchToTorus32(-1, 4*m-4);
    lweNoiselessTrivial(temp2,modSwitchToTorus32(1,4*m-4),params);
    for(int i=0;i<10;i++){
        lweNoiselessTrivial(temp_result, AndConst, bootk->bkFFT->in_out_params);
        lweNoiselessTrivial(label,modSwitchToTorus32(labels[i]==0?-1:1,4*m-4),params);
        lweAddTo(temp_result, label, bootk->bkFFT->in_out_params);
        lweAddTo(temp_result, deltaVector[i], bootk->bkFFT->in_out_params);
        lweCopy(classVector[i],temp_result,bootk->bkFFT->in_out_params);
        tfhe_bootstrap_woKS_FFT(classVector[i],bootk->bkFFT, modSwitchToTorus32(1, 4*m-4),temp_result);
        lweNoiselessTrivial(temp,modSwitchToTorus32(1,4*m-4),params);
        lweAddTo(classVector[i], temp, bootk->bkFFT->in_out_params);
        
        lweAddTo(deltaVector[i], temp2, bootk->bkFFT->in_out_params);
    }
    
 /*   //sommer le vecteur delta
   LweSample *sum=new_LweSample(bootk->bkFFT->in_out_params);
    lweClear(sum,params);
    for(int i=0;i<d;i++){
        lweAddTo(sum,deltaVector[i],params);
    }
    //calculer sign(K/2-NB) si positif alors la classe majoritaire est la classe negative sinon classe positif
    lweNoiselessTrivial(predictedClass,modSwitchToTorus32(k/2,4*m-4),params);
    //lweSubTo(sum,predictedClass,params);
    //tfhe_bootstrap_FFT(predictedClass,bootk->bkFFT,modSwitchToTorus32(1,8),sum);
    //lweNegate(predictedClass,predictedClass,params);
    lweCopy(predictedClass,deltaVector[0],params);*/
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC; 
    cout << "Temps scoring operations " <<  time_spent <<"secondes"<< endl;
    
}


void KNN2(LweSample *predictedClass,float dataset[NBLINE][NBCOL],int labels[NBLINE],LweSample *M[NBCOL],const LweParams *params, TFheGateBootstrappingCloudKeySet *bootk,int d,int m,int k,int nb_col){
    /**************************************************************************************/
    /************************* Constrction de delta matrices ******************************/
    if (VERBOSE) cout << "Construction la matrice des delta values"<<endl;
    
    LweSample *deltaValues[NBLINE][NBLINE];
    for(int i=0;i<d;i++){
        for(int j=0;j<d;j++){
            deltaValues[i][j]=new_LweSample(params);
        }
    }
    LweSample *deltaVector[NBLINE];
    for(int i=0;i<d;i++){
        deltaVector[i]=new_LweSample(params);
    }
    LweSample *classVector[NBLINE];
    for(int i=0;i<d;i++){
        classVector[i]=new_LweSample(params);
    }
clock_t begin = clock();
    deltaValuesMatrix(deltaValues,dataset,M, params,bootk->bkFFT,m,d,nb_col);
   
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC; 
    if (VERBOSE && TEST) cout << "Temps construction de deltamatrices" <<  time_spent<< endl;
    
    
    /**************************************************************************************/
    /******************************* Scoring operation ************************************/
    
    cout <<endl <<"Scoring operation   "<< endl ;
    
    begin = clock();

    scoreOperation(deltaVector,deltaValues,bootk->bkFFT,bootk->bkFFT->extract_params,d,m,k);

    LweSample *label=new_LweSample(bootk->bkFFT->in_out_params);
    LweSample *temp=new_LweSample(bootk->bkFFT->in_out_params);
    LweSample *temp2=new_LweSample(bootk->bkFFT->in_out_params);
    //Appliquer AND operation with between deltavector and datasetlabels
    LweSample *temp_result = new_LweSample(bootk->bkFFT->in_out_params);
    static const Torus32 AndConst = modSwitchToTorus32(-1, 4*m-4);
    lweNoiselessTrivial(temp2,modSwitchToTorus32(1,4*m-4),params);
    for(int i=0;i<10;i++){
        lweNoiselessTrivial(temp_result, AndConst, bootk->bkFFT->in_out_params);
        lweNoiselessTrivial(label,modSwitchToTorus32(labels[i]==0?-1:1,4*m-4),params);
        lweAddTo(temp_result, label, bootk->bkFFT->in_out_params);
        lweAddTo(temp_result, deltaVector[i], bootk->bkFFT->in_out_params);
        lweCopy(classVector[i],temp_result,bootk->bkFFT->in_out_params);
        tfhe_bootstrap_woKS_FFT(classVector[i],bootk->bkFFT, modSwitchToTorus32(1, 4*m-4),temp_result);
        lweNoiselessTrivial(temp,modSwitchToTorus32(1,4*m-4),params);
        lweAddTo(classVector[i], temp, bootk->bkFFT->in_out_params);
        
        lweAddTo(deltaVector[i], temp2, bootk->bkFFT->in_out_params);
    }
    
    //sommer le vecteur delta
    LweSample *sum=new_LweSample(bootk->bkFFT->in_out_params);
    lweClear(sum,params);
    for(int i=0;i<d;i++){
        lweAddTo(sum,classVector[i],params);
    }
    //lweCopy(predictedClass, sum,params);
    //calculer sign(K/2-NB) si positif alors la classe majoritaire est la classe negative sinon classe positif
    lweNoiselessTrivial(predictedClass,modSwitchToTorus32(k/2,2*m-2),params);
    lweSubTo(sum,predictedClass,params);
    tfhe_bootstrap_FFT(predictedClass,bootk->bkFFT,modSwitchToTorus32(1,8),sum);
    //lweNegate(predictedClass,predictedClass,params);
    //lweCopy(predictedClass,sum,params);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC; 
    cout << "Temps scoring operations " <<  time_spent <<"secondes"<< endl;
    
}
/**
 * @brief 
 * 
 * @return int 
 */
int main(int argc, char** argv){
    if (argc<2){
        cout << "Vous devez indiquer le nombre de données dans ce dataset"<<endl;
        return -1;
    }       
    
    /******************************************************************************/
    /************************ Initialisation des Parametres ***********************/
    if (VERBOSE) cout << "Initialisation des parametres "<<endl;
    int32_t N=1024;
    int32_t tau=100;
    int32_t v=4;
    double alpha=1./(double)pow(10,9);
    int nb_col=atoi(argv[4]);
    int32_t d=atoi(argv[1]);
    int32_t m=atoi(argv[2]);
    int32_t k=atoi(argv[3]);

    char cmd[512] ="python3 entrainmentKNN.py ";
    char d_car[6];
    sprintf(d_car, "%d ", d);
    strcat(cmd, d_car );
    sprintf(d_car, "%d ", k);
    system(strcat(cmd, d_car ));
    
    /******************************************************************************/
    /******************Generer des clés********************************************/
    clock_t begin = clock();



    if (VERBOSE) cout << "Géneration des clés: "<<endl;
    if (VERBOSE) cout << "------- TLWE Key: "<<endl;
    TLweParams *params = new_TLweParams(1024,1,0,alpha);
    TLweKey *key1024_1 =new_TLweKey(params);


    if (VERBOSE) cout << "------- LWE Key: "<<endl;
    LweParams *paramsLwe=new_LweParams(1024,0,alpha);
    LweKey *key_lwe = new_LweKey(paramsLwe);
    tLweExtractKey(key_lwe,key1024_1);

    if (VERBOSE) cout << "------- Bootstrapping Key: "<<endl;
    TGswParams *params_tgsw=new_TGswParams(6,64,params);
    TGswKey *tgsw_key = new_TGswKey(params_tgsw);
    tGswKeyGen(tgsw_key);

    LweBootstrappingKey *bk = new_LweBootstrappingKey(18, 1, paramsLwe,
                                                      params_tgsw);
    tfhe_createLweBootstrappingKey(bk, key_lwe, tgsw_key);
    LweBootstrappingKeyFFT *bs_key = new_LweBootstrappingKeyFFT(bk);
   
    clock_t end = clock();
double time_spent = (double)(end - begin) / CLOCKS_PER_SEC; 
cout << "Temps generation des clés  " <<  time_spent << "secondes" << endl <<endl;
const TFheGateBootstrappingParameterSet *paramsGate=new TFheGateBootstrappingParameterSet(8,2,bk->in_out_params,params_tgsw);
TFheGateBootstrappingCloudKeySet *bootk=new TFheGateBootstrappingCloudKeySet(paramsGate, bk, bs_key);




    /**************************************************************************************/
    /*******************************Lecture de dataset ************************************/
    begin=clock();
    if (VERBOSE) cout << "Lecture de dataset: "<<endl;
    //float vecSrc[30]={0.521037,0.022658,0.545989,0.363733,0.593753,0.792037,0.703140,0.731113,0.686364,0.605518,0.356147,0.120469,0.369034,0.273811,0.159296,0.351398,0.135682,0.300625,0.311645,0.183042,0.620776,0.141525,0.668310,0.450698,0.601136,0.619292,0.568610,0.912027,0.598462,0.418864};
    float vecSrc[30]={0.273984, 0.395671, 0.264184, 0.154358, 0.314706 ,0.143028, 0.072915, 0.142346, 0.320202, 0.271904, 0.224371, 0.306710, 0.205485, 0.087501, 0.097155 ,0.117523 ,0.054949 ,0.332828, 0.363708, 0.172056, 0.207044 ,0.305970, 0.192390, 0.096908, 0.149970, 0.060628, 0.041422, 0.164021, 0.121033 ,0.089663};
    //float vecSrc[30]={0.472222, 0.083333, 0.677966, 0.583333};
    float dataset[NBLINE][NBCOL];
    int classes[NBLINE];
    lireDataSet(dataset,classes,d,nb_col);

     end = clock();
        time_spent = (double)(end - begin) / CLOCKS_PER_SEC; 
    cout << "Temps Lecture de dataset  " <<  time_spent  <<"secondes" << endl;

    
    
    
    
    /***************************************************************************************/
    /************************* Chiffrer les données necessaire *****************************/
    if (VERBOSE) cout << "------- Chiffrer le M"<<endl;
    //chiffer le dataset (les Ci)
    LweSample *MEncrypted[NBCOL];
    Torus32 Mu;
    begin 
    for (int i =0;i<NBCOL;i++){
        MEncrypted[i]=new_LweSample(paramsLwe);
        Mu=dtot32(vecSrc[i]/400.);
        lweSymEncrypt(MEncrypted[i], Mu, alpha,key_lwe);
    }   
    
    
        end = clock();
        time_spent = (double)(end - begin) / CLOCKS_PER_SEC; 
    cout << "Temps Chiffrement " <<  time_spent  <<"secondes" << endl;
        
    



/****************************************************************************************/
/**************************** KNN Function ******************************************/
    LweSample *deltaVector[NBLINE];
    for(int i=0;i<d;i++){
        deltaVector[i]=new_LweSample(paramsLwe);
    }
    LweSample *classVector[NBLINE];
    for(int i=0;i<d;i++){
        classVector[i]=new_LweSample(paramsLwe);
    }
    begin =clock();
    LweSample *predictedClass=new_LweSample(paramsLwe);
   // KNN(deltaVector,classVector,dataset,classes,MEncrypted,paramsLwe,bootk,d,m,k,nb_col);
   double debut=omp_get_wtime();
    KNN2(predictedClass,dataset,classes,MEncrypted,paramsLwe,bootk,d,m,k,nb_col);
double fin=omp_get_wtime();
    end=clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC; 
    cout <<endl<< "Temps pour effectuer un KNN " <<  fin-debut<< "secondes" <<endl;
    FILE* timeExec = NULL; 
    timeExec = fopen("timeExecution.data","a+");
   
    if (timeExec != NULL )
    {
        fprintf(timeExec,"%f\n", fin-debut);
        fclose(timeExec); 
        
    }
    else
    {
        printf("Impossible d'ouvrir le fichier timeExection");
    }

    //verification de delta vector
    Torus32 predicted;
    predicted=lweSymDecrypt(predictedClass,key_lwe,8);
    cout << "Class predite " <<" "<<t32tod(predicted)*8<<endl;
    /*Torus32 deltavectorValue;
    Torus32 classValue;
    for(int i=0;i<d;i++){
        deltavectorValue=lweSymDecrypt(deltaVector[i],key_lwe,4*k);
        classValue=lweSymDecrypt(classVector[i],key_lwe,4*m-4);
        cout << "delta vector chiffrée " << i<< " "<<t32tod(deltavectorValue)<< "  classes " << t32tod(classValue)<<endl;
    }*/
    
    

   

    
    
    
   
   
cout <<endl<<endl << "TEST" <<endl;
/*for (int i=0;i<NBLINE;i++){
    for (int j=0;j<NBLINE;j++){
    Torus32 distance_cipher=lweSymDecrypt(deltaValues[i][j], key_lwe, MSIZE);
    float distance=t32tod(distance_cipher);    

    cout <<"Mauvaise Distance " << i<<j<<"   "<<distance <<endl;
}
}*/
    
    //decrypt
      cout << "[";
for(int j=0;j<d;j++){
    for (int k=0;k<d;k++){
        
    //cout << distance << ", " ;
    //cout <<"Mauvaise Distance " << j<<k<<"   "<<distance << "          ";
    //calculer di
    float d1=0;
   for(int i=0;i<nb_col;i++){
        d1+=pow(dataset[j][i]-vecSrc[i],2);
    }
//calculer dj
    float d2=0;
   for(int i=0;i<nb_col;i++){
        d2+=pow(dataset[k][i]-vecSrc[i],2);
    }
    
   // cout << (d1-d2)/16 << ", " ;
//    if(k==d-1)cout <<endl;
if (j==0)cout <<d2 << " ,";
    //cout <<"Vrai Distance "<< j<<k<<"   " <<((d1-d2)/9)<< endl;
}
}


/*
float result=formule2Clair(dataset[0], dataset[1], vecSrc,nb_col);
cout << (result)<<endl;

LweSample *resultC=new_LweSample(paramsLwe);float Ai_vector[NBCOL];
begin = clock();
formule2(resultC, dataset[0], dataset[1],Ai_vector, MEncrypted,paramsLwe,nb_col);
end=clock();
time_spent = (double)(end - begin) ; 
cout <<endl<< "Temps pour effectuer un KNN " <<  time_spent<< "secondes" <<endl;
Torus32 delta;
delta=lweSymDecrypt(resultC,key_lwe,4*m-4);
cout << "delta vector chiffrée " << " "<<t32tod(delta)<<endl;
    */

    return 0;
}

