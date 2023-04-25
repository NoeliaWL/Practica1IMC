//============================================================================
// Introduction to computational models
// Name        : la1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>    // To obtain current time time()
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <string.h>
#include <math.h>
#include <sstream>

#include "imc/MultilayerPerceptron.h"
#include "imc/util.h"


using namespace imc;
using namespace std;
using namespace util;

int main(int argc, char **argv) {
    // Process arguments of the command line
    bool tflag=0, Tflag = 0, iflag=0, lflag=0, hflag=0, eflag=0, mflag=0, sflag=0, wflag = 0, pflag = 0;
    char *tvalue = NULL, *Tvalue = NULL, *ivalue = NULL, *lvalue = NULL, *hvalue = NULL, *evalue = NULL, *mvalue = NULL, *wvalue = NULL;
    int c;

    opterr = 0;

    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "t:T:i:l:h:e:m:w:ps")) != -1)
    {
        // The parameters needed for using the optional prediction mode of Kaggle have been included.
        // You should add the rest of parameters needed for the lab assignment.
        switch(c){
            case 't':
                tflag = true;
                tvalue = optarg;
                break;
            case 'T':
                Tflag = true;
                Tvalue = optarg;
                break;
            case 'i':
                iflag = true;
                ivalue = optarg;
                break;
            case 'l':
                lflag = true;
                lvalue = optarg;
                break;
            case 'h':
                hflag = true;
                hvalue = optarg;
                break;
            case 'e':
                eflag = true;
                evalue = optarg;
                break;
            case 'm':
                mflag = true;
                mvalue = optarg;
                break;
            case 's':
                sflag = true;
                break;
            case 'w':
                wflag = true;
                wvalue = optarg;
                break;
            case 'p':
                pflag = true;
                break;
            case '?':
                if (optopt == 'T' || optopt == 'w' || optopt == 'p')
                    fprintf (stderr, "The option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                             "Unknown character `\\x%x'.\n",
                             optopt);
                return EXIT_FAILURE;
            default:
                return EXIT_FAILURE;
        }
    }

    if (!pflag) {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        // Multilayer perceptron object
    	MultilayerPerceptron mlp;

        // Parameters of the mlp. For example, mlp.eta = value;
        if(eflag) { mlp.eta = atof(evalue); }
        if(mflag) { mlp.mu = atof(mvalue); }
    	int iterations = iflag ? atoi(ivalue) : 1000;

        // Read training and test data: call to util::readData(...)
    	Dataset * trainDataset = readData(tvalue);
    	Dataset * testDataset = readData(Tflag ? Tvalue : tvalue);

        // Normalize the data
        if(sflag){
            minMaxScalerDataSetInputs(trainDataset, -1.0, 1.0, minDatasetInputs(trainDataset), maxDatasetInputs(trainDataset));
            minMaxScalerDataSetOutputs(trainDataset, 0.0, 1.0, minDatasetOutputs(trainDataset), maxDatasetOutputs(trainDataset));

            minMaxScalerDataSetInputs(testDataset, -1.0, 1.0, minDatasetInputs(trainDataset), maxDatasetInputs(trainDataset));
            minMaxScalerDataSetOutputs(testDataset, 0.0, 1.0, minDatasetOutputs(trainDataset), maxDatasetOutputs(trainDataset));
        }

        // Initialize topology vector
    	int layers = lflag ? atoi(lvalue) : 1;
    	int * topology = new int[layers + 2];
        int neuronsHide = hflag ? atoi(hvalue) : 5;

        topology[0] = trainDataset->nOfInputs;
        topology[layers + 1] = testDataset->nOfOutputs;
        for(int i=1; i<layers+1; i++){
            topology[i] = neuronsHide;
        }

        // Initialize the network using the topology vector
        mlp.initialize(layers+2,topology);

        // Seed for random numbers
        int seeds[] = {1,2,3,4,5};
        double *testErrors = new double[5];
        double *trainErrors = new double[5];
        double bestTestError = 1;
        string name;
        for(int i=0; i<5; i++){
            cout << "**********" << endl;
            cout << "SEED " << seeds[i] << endl;
            cout << "**********" << endl;
            srand(seeds[i]);
            ostringstream auxName;
            auxName << "seed_" << i << ".txt";
            name = auxName.str();
            mlp.runOnlineBackPropagation(trainDataset,testDataset,iterations,&(trainErrors[i]),&(testErrors[i]), name);
            cout << "We end!! => Final test error: " << testErrors[i] << endl;

            // We save the weights every time we find a better model
            if(wflag && testErrors[i] <= bestTestError)
            {
                mlp.saveWeights(wvalue);
                bestTestError = testErrors[i];
            }
        }

        cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

        double averageTestError = 0, stdTestError = 0;
        double averageTrainError = 0, stdTrainError = 0;
        
        // Obtain training and test averages and standard deviations
        for(int i=0; i<5; i++){
            averageTrainError += trainErrors[i];
            averageTestError += testErrors[i];
        }
        averageTrainError /= 5;
        averageTestError /= 5;

        double varianzaTrain = 0.0, varianzaTest = 0.0;

        for(int i=0; i<5; i++){
            double rangoTrain = 0.0, rangoTest = 0.0;
            rangoTrain = pow(trainErrors[i] - averageTrainError, 2);
            rangoTest = pow(testErrors[i] - averageTestError, 2);
            varianzaTrain += rangoTrain;
            varianzaTest += rangoTest;
        }
        varianzaTrain /= 5;
        varianzaTest /= 5;

        stdTrainError = sqrt(varianzaTrain);
        stdTestError = sqrt(varianzaTest);

        cout << "FINAL REPORT" << endl;
        cout << "************" << endl;
        cout << "Train error (Mean +- SD): " << averageTrainError << " +- " << stdTrainError << endl;
        cout << "Test error (Mean +- SD):          " << averageTestError << " +- " << stdTestError << endl;
        return EXIT_SUCCESS;
    }
    else {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////
        
        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Initializing the network with the topology vector
        if(!wflag || !mlp.readWeights(wvalue))
        {
            cerr << "Error while reading weights, we can not continue" << endl;
            exit(-1);
        }

        // Reading training and test data: call to util::readData(...)
        Dataset *testDataset;
        testDataset = readData(Tvalue);
        if(testDataset == NULL)
        {
            cerr << "The test file is not valid, we can not continue" << endl;
            exit(-1);
        }

        mlp.predict(testDataset);

        return EXIT_SUCCESS;
    }

    
}

