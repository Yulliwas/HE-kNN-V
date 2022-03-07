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
void lireDataSet(float dataset[NBLINE][NBCOL], char index[NBLINE],int d,int nb_col){
    FILE* data = NULL; FILE* label = NULL;
    data = fopen("data.data","r+");
    label = fopen("label.data","r+");
    if (data != NULL && label != NULL)
    {
        for(int i=0;i<d;i++){
            fscanf(label,"%c",&index[i]);
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

        tfhe_MuxRotate_FFT(temp2, temp3, bkFFT + i, barai, bk_params);
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
void formule2(TLweSample *result, TLweSample *Ai,TLweSample *Aj,TLweSample *Ci, TLweSample *Cj, IntPolynomial *M,const TLweParams *params,int nb_col){
    TLweSample *Ai_Aj=new_TLweSample(params);
    tLweCopy(Ai_Aj,Ai,params);
    tLweSubTo(Ai_Aj, Aj, params);

    TLweSample *Cj_Ci=new_TLweSample(params);
    tLweCopy(Cj_Ci,Cj,params);
    tLweSubTo(Cj_Ci, Ci, params);

    tLweCopy(result,Cj_Ci,params);
    IntPolynomial *Mx2=new_IntPolynomial(params->N);
    //Laforme de dataset est connu on connait le nombre de colonne
    intPolynomialClear(Mx2);
    for (int i=0;i<nb_col;i++){
        Mx2->coefs[i]=2*M->coefs[i];
       
    }
    

    multiplyPolynomial(result, Cj_Ci,Mx2, params);

    tLweAddTo(result,Ai_Aj,params);
}

void formule2Clair(TorusPolynomial *result, TorusPolynomial *Ai,TorusPolynomial *Aj,TorusPolynomial *Ci, TorusPolynomial *Cj, IntPolynomial *M,int nb_col){
    int params=1024;
    TorusPolynomial *Ai_Aj=new_TorusPolynomial(params);
    torusPolynomialCopy(Ai_Aj,Ai);
    torusPolynomialSubTo(Ai_Aj, Aj);

    TorusPolynomial *Cj_Ci=new_TorusPolynomial(params);
    torusPolynomialCopy(Cj_Ci,Cj);
    torusPolynomialSubTo(Cj_Ci, Ci);

  
    IntPolynomial *Mx2=new_IntPolynomial(params);

    intPolynomialClear(Mx2);
    for (int i=0;i<nb_col;i++){
        Mx2->coefs[i]=2*M->coefs[i];
        //cout << "2m  "<< Mx2->coefs[0] << "  " << M->coefs[0]<<endl;
    }
    

    torusPolynomialMultKaratsuba(result, Mx2,Cj_Ci);

    torusPolynomialAddTo(result,Ai_Aj);
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
void deltaValueUnique(LweSample *deltaValue, TLweSample *Ai,TLweSample *Aj,TLweSample *Ci, TLweSample *Cj, IntPolynomial *M,const TLweParams *params,const LweBootstrappingKeyFFT *bs_key,int m,int nb_col){
    TLweSample *formula2 = new_TLweSample(params);
    tLweClear(formula2,params);
    clock_t  begin=clock();
    formule2(formula2, Ai,Aj,Ci,Cj, M,params,nb_col);
    clock_t  end=clock();
    cout << "Temps formule " <<  (end-begin) << "secondes" << endl <<endl;
    LweSample *result = new_LweSample(&params->extracted_lweparams);
    tLweExtractLweSampleIndex(result, formula2, nb_col-1, &params->extracted_lweparams,  params) ;
     
    //do a sign bootstrapping 
    int32_t b_delta=(4*m/*-4*/);//4m-4 parameter dans Zuber
    Torus32 MU = modSwitchToTorus32(1,b_delta);
    begin=clock();
    tfhe_bootstrap_woKS_FFT(deltaValue,bs_key,MU,result);
    end=clock();
    cout << "Temps bootstrap " <<  (end-begin) << "secondes" << endl <<endl;
   // lweCopy(deltaValue,result,&params->extracted_lweparams);
    //convert to 0 or 1
    LweSample *toAdd=new_LweSample(&params->extracted_lweparams);
    lweNoiselessTrivial(toAdd,  MU, &params->extracted_lweparams);
    begin=clock();
    lweAddTo(deltaValue,toAdd,&params->extracted_lweparams);
    end=clock();
    cout << "Temps addition " <<  (end-begin) << "secondes" << endl <<endl;
}


/**
 * @brief Construire la matrice deltaValue (Marche tres bien)
 * 
 * @param datasetEncoded 
 * @param datasetEncrypted 
 * @param alpha 
 * @param key1024_1 
 */
void deltaValuesMatrix(LweSample *deltaValues[NBLINE][NBLINE],TLweSample *Ai_vect[NBLINE],TLweSample *datasetEncrypted[NBLINE],IntPolynomial *M,  const TLweParams *params,const LweBootstrappingKeyFFT *bs_key,int m,int d,int nb_col){
    const LweParams *in_out_params = bs_key->in_out_params;
    static const Torus32 MU = modSwitchToTorus32(1,2*m);
    for(int i=0;i<d;i++){
        lweClear(deltaValues[i][i],&params->extracted_lweparams);
        
        for(int j=i+1;j<d;j++){
            
            //cout << " i j "<<i << j <<endl;
            deltaValueUnique(deltaValues[i][j],Ai_vect[i],Ai_vect[j],datasetEncrypted[i],datasetEncrypted[j],M,params,bs_key,m,nb_col);
            //dji=1-dij
            lweNoiselessTrivial(deltaValues[j][i],  MU, in_out_params);
            lweSubTo(deltaValues[j][i],deltaValues[i][j],&params->extracted_lweparams);
        }
        
    }
   
}

void scoreBlock(LweSample *deltaVectorValue,LweSample *sum,LweBootstrappingKeyFFT *bk,const LweParams *params,int m,int16_t k){
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

void scoreOperation(LweSample *deltaVector[NBLINE],LweSample *deltaValues[NBLINE][NBLINE],LweBootstrappingKeyFFT *bk,const LweParams *params,int d,int m,int16_t k){
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
       homomorphicSign(result,bk,modSwitchToTorus32(1,4),deltaVector[i],k,m);
      // MU2 = modSwitchToTorus32(1,4);
        lweNoiselessTrivial(toAdd,  modSwitchToTorus32(1,4), &bk->accum_params->extracted_lweparams);
        for(int i=0;i<k;i++){
        
            lweNegate(result[i], result[i], &bk->accum_params->extracted_lweparams);
            
            lweAddTo(result[i],toAdd,&bk->accum_params->extracted_lweparams);
            
        }
        lweCopy(deltaVector[i],result[k-1],params); 
        
        
    }
    
    
}

bool approxEquals(float a, float b) { return abs(a - b) < 0.01; }
bool signEgal(double a,double b){
    if ((a>=0 && b>=0) || (a<0 && b<0)) return true;
    return false;
}


void scoreOperation2(LweSample *deltaVector[NBLINE],LweSample *deltaValues[NBLINE][NBLINE],LweBootstrappingKeyFFT *bk,const LweParams *params,int d,int m,int16_t k){
    //variables a utiliser 
    LweSample *result[NBRESULT];
    LweSample *sum=new_LweSample(params);
    int compteBlock;
    int cpt;
    int compteurAddition;
    bool premierBloc=true;

    LweSample *toSub=new_LweSample(&bk->accum_params->extracted_lweparams);
    Torus32 MU2=modSwitchToTorus32(m,2*m-2);
    lweNoiselessTrivial(toSub,MU2,&bk->accum_params->extracted_lweparams); 
    LweSample *toAdd=new_LweSample(&bk->accum_params->extracted_lweparams);
    Torus32 MU=modSwitchToTorus32(k,2*m-2);
    lweNoiselessTrivial(toAdd,MU,&bk->accum_params->extracted_lweparams); 

    int m_size=2*m-2;
    
    for(int i=0;i<NBRESULT;i++){
        result[i]=new_LweSample(params);
    }

        
    
    
    
    //partie à faire 
    for(int i=0;i<d;i++){
        for(int j=0;j<d;j++){
            if (j==m){
                lweSubTo(deltaVector[i],toSub,params);
                tfhe_bootstrap_woKS_FFT(deltaVector[i],bk,modSwitchToTorus32(1,2*m-2),deltaVector[i]);
            }
            if (j==i){
                continue;
            } 

            
            //cout << "ici "<< i<<j <<"   "<<compteurAddition<<endl;
            //effectuer les additions
            
            lweAddTo(deltaVector[i],deltaValues[j][i],params);
            
            

            

           
            
        }
        lweAddTo(deltaVector[i],toAdd,params);
       //tfhe_bootstrap_woKS_FFT(deltaVector[i],bk,modSwitchToTorus32(1,2*m-2),deltaVector[i]);
        

    }
    
    
}




void KNN(LweSample *deltaVector[NBLINE],TLweSample *datasetEncrypted[NBLINE],TLweSample *Ai_vect[NBLINE],IntPolynomial* M,const TLweParams *params, LweBootstrappingKeyFFT *bs_key,LweParams *paramsLwe,int d,int m,int k,int nb_col){
    /**************************************************************************************/
    /************************* Constrction de delta matrices ******************************/
    if (VERBOSE) cout << "Construction la matrice des delta values"<<endl;
    clock_t begin = clock();
    LweSample *deltaValues[NBLINE][NBLINE];
    for(int i=0;i<d;i++){
        for(int j=0;j<d;j++){
            deltaValues[i][j]=new_LweSample(paramsLwe);
        }
    }
    deltaValuesMatrix(deltaValues,Ai_vect,datasetEncrypted,M, params,bs_key,m,d,nb_col);
   
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC; 
    if (VERBOSE && TEST) cout << "Temps construction de deltamatrices" <<  time_spent<< endl;
    
    
    /**************************************************************************************/
    /******************************* Scoring operation ************************************/
    cout <<endl <<"Scoring operation   "<< endl ;
    
    begin = clock();

    scoreOperation2(deltaVector,deltaValues,bs_key,bs_key->extract_params,d,m,k);

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
    sprintf(d_car, "%d", k);
    system(strcat(cmd, d_car ));
    
    /******************************************************************************/
    /******************Generer des clés********************************************/
    clock_t begin = clock();



    if (VERBOSE) cout << "Géneration des clés: "<<endl;
    if (VERBOSE) cout << "------- TLWE Key: "<<endl;
    const TLweParams *params = new_TLweParams(1024,1,0,alpha);
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




    /**************************************************************************************/
    /*******************************Lecture de dataset ************************************/
    begin=clock();
    if (VERBOSE) cout << "Lecture de dataset: "<<endl;
    float vecSrc[30]={0.521037,0.022658,0.545989,0.363733,0.593753,0.792037,0.703140,0.731113,0.686364,0.605518,0.356147,0.120469,0.369034,0.273811,0.159296,0.351398,0.135682,0.300625,0.311645,0.183042,0.620776,0.141525,0.668310,0.450698,0.601136,0.619292,0.568610,0.912027,0.598462,0.418864};
    
    float dataset[NBLINE][NBCOL];
    char index[NBLINE];
    lireDataSet(dataset,index,d,nb_col);

     end = clock();
        time_spent = (double)(end - begin) / CLOCKS_PER_SEC; 
    cout << "Temps Lecture de dataset  " <<  time_spent  <<"secondes" << endl;

    /**************************************************************************************/
    /********************* Encodage des données dans des polynomes ************************/
    begin=clock();
    if (VERBOSE) cout << "Encodage des données dans des polynomes"<<endl;
    IntPolynomial* M=vectorSrcToIntPolynomial(vecSrc,N,nb_col,v,tau);
    TorusPolynomial *datasetEncoded[NBLINE];
    //encoder le dataset 
    encodeDataset(dataset, datasetEncoded,N,d,nb_col,v,tau);

     end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC; 
    cout << "Temps Encodage de dataset " <<  time_spent << "secondes"<< endl;
    //Verification
    if (VERBOSE)
    for(int i=0;i<d;i++){
        for(int j=0;j<nb_col;j++){
            if (!approxEquals(dataset[i][j],(t32tod(datasetEncoded[i]->coefsT[j])*400))) cout << "Valeur en clair "<< dataset[i][j] << "         Valeur encodé "<<t32tod(datasetEncoded[i]->coefsT[j])*300<<endl;
        }
    }
   
    
    /***************************************************************************************/
    /************************* Chiffrer les données necessaire *****************************/
    if (VERBOSE) cout << "Chiffrement des données encodés"<<endl;
    if (VERBOSE) cout << "------- Chiffrer les Ai"<<endl;
    //chiffrer les Ai
    begin=clock();
    TLweSample *Ai_vect[NBLINE];
    for(int i=0;i<d;i++)
        Ai_vect[i]=new_TLweSample(params);
    calculerAi_vect(dataset,Ai_vect, N,v,alpha,key1024_1,d,nb_col);
    //Verification des Ai
    for(int i=0;i<d;i++){
        TorusPolynomial *result=new_TorusPolynomial(N);
        tLweSymDecrypt(result,Ai_vect[i], key1024_1, MSIZE);
        if (!approxEquals(t32tod(calculerAi(dataset[i],N,v,nb_col)->coefsT[nb_col-1]),t32tod(result->coefsT[nb_col-1]))) 
            cout << "Valeur Ai en clair "<< t32tod(calculerAi(dataset[i],N,v,nb_col)->coefsT[nb_col-1]) << "         Valeur encrypté "<<t32tod(result->coefsT[nb_col-1])<<endl;
    }



    if (VERBOSE) cout << "------- Chiffrer les Ci (dataset)"<<endl;
    //chiffer le dataset (les Ci)
    TLweSample *datasetEncrypted[NBLINE];
    for(int i=0;i<d;i++)
        datasetEncrypted[i]=new_TLweSample(params);
    encrypt_dataset(datasetEncoded,datasetEncrypted,alpha,  key1024_1,d);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC; 
    cout << "Temps Chiffrer de dataset " <<  time_spent << "secondes"<< endl;
    //verification des Ci
    TorusPolynomial *result2=new_TorusPolynomial(N);
    for(int i=0;i<d;i++){
        
        torusPolynomialClear(result2);
        tLweSymDecrypt(result2,datasetEncrypted[i], key1024_1, MSIZE);
        for(int j=0;j<nb_col;j++){
            if (!approxEquals(t32tod(datasetEncoded[i]->coefsT[j]),t32tod(result2->coefsT[j]))) 
            cout << "Valeur Ci en clair "<< i << j << "  "<< t32tod(datasetEncoded[i]->coefsT[j]) << "         Valeur encrypté "<<t32tod(result2->coefsT[j])<<endl;
        }
    }
    



/****************************************************************************************/
/**************************** KNN Function ******************************************/
    begin =clock();
    LweSample *deltaVector[NBLINE];
    for(int i=0;i<d;i++)deltaVector[i]=new_LweSample(paramsLwe);
    KNN(deltaVector,datasetEncrypted,Ai_vect,M,params,bs_key,paramsLwe,d,m,k,nb_col);

    end=clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC; 
    cout <<endl<< "Temps pour effectuer un KNN " <<  time_spent<< "secondes" <<endl;
    FILE* timeExec = NULL; 
    timeExec = fopen("timeExecution.data","a+");
   
    if (timeExec != NULL )
    {
        fprintf(timeExec,"%f\n",time_spent);
        fclose(timeExec); 
        
    }
    else
    {
        printf("Impossible d'ouvrir le fichier timeExection");
    }

    //verification de delta vector
    Torus32 deltavectorValue;
    for(int i=0;i<d;i++){
        deltavectorValue=lweSymDecrypt(deltaVector[i],key_lwe,4*m-4);
        cout << "delta vector chiffrée " << i<< " "<<t32tod(deltavectorValue)<<endl;
    }

   

    
    
    
   
   
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
    
  //  cout << (d1-d2)/16 << ", " ;
//    if(k==d-1)cout <<endl;
if (j==0)cout <<d2 << " ,";
    //cout <<"Vrai Distance "<< j<<k<<"   " <<((d1-d2)/9)<< endl;
}
}
cout << endl;

    return 0;
}

