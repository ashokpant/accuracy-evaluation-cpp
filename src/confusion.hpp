//
// Created by Ashok K. Pant on 3/17/16.
//

#ifndef ACCURACY_EVALUATION_CPP_CONFUSION_HPP
#define ACCURACY_EVALUATION_CPP_CONFUSION_HPP

#include <iostream>
#include <string>
#include <vector>
#include <tgmath.h> //round,roundf
#include <sstream>  // std::stringstream, std::stringbuf
#include <algorithm>
#include <array>
#include <set>

using namespace std;

class Confusion {
public:
    int _classes;
    int _samples;
    double _c;
    vector<vector<double>> _per; //per[classes][4]
    vector<vector<string>> _ind; //ind[classes][classes]
    vector<vector<int>> _cm; //cm[classes][classes]

    Confusion(int classes, int samples) : _classes(classes), _samples(samples),
                                          _per(classes, vector<double>(4)), _ind(classes, vector<string>(classes)),
                                          _cm(classes, vector<int>(classes)) { }

    Confusion(vector<vector<double>> targets, vector<vector<double>> outputs) {
        confusion(targets, outputs);
    }

    Confusion(vector<int> targets, vector<int> outputs) {
        vector<vector<double>> tar;
        vector<vector<double>> out;
        convertToBooleanMatrix(targets, outputs, tar, out);
        confusion(tar, out);
    }

    unsigned calcNumClasses( const vector<int> & v1, const vector<int> & v2 )
    {
        std::set<int> all{ v1.cbegin(), v1.cend() };
        all.insert( v2.cbegin(), v2.cend() );
        return all.size();
    }

    void convertToBooleanMatrix(vector<int> targets, vector<int> outputs, vector<vector<double>> &tar,
                                vector<vector<double>> &out) {
        int numClasses = calcNumClasses( targets, outputs );
        int numSamples = targets.size();
        vector<vector<double>> t(numClasses, vector<double>(numSamples));
        vector<vector<double>> o(numClasses, vector<double>(numSamples));
        for (int i = 0; i < numSamples; ++i) {
            t[targets.at(i)][i] = 1;
            o[outputs.at(i)][i] = 1;
        }
        tar = t;
        out = o;
    }

    void confusion(vector<vector<double>> targets, vector<vector<double>> outputs) {
        /* confusion takes an SxQ (S:Classes; Q:Samples)target and output matrices
             T and Y, where each column of T is all zeros with one 1 indicating the target
             class, and where the columns of Y have values in the range [0,1], the largest
             Y indicating the models output class.

            It returns the confusion value C, indicating the fraction of samples
            misclassified, CM an SxS confusion matrix, where CM(i,j) is the number
            of target samples of the ith class classified by the outputs as class j.

            IND is an SxS cell array whose elements IND{i,j} contain the sample
            indices of class i targets classified as class j.

            PER is an Sx4 matrix where each ith row summarizes these percentages
            associated with the ith class:
                S(i,1) = false negative rate = false negatives / all output negatives
                S(i,2) = false positive rate = false positives / all output positives
                S(i,3) = true positive rate = true positives / all output positives
                S(i,4) = true negative rate = true negatives / all output negatives
               */

        int numClasses = targets.size(); //row= classes , col=samples

        if (numClasses == 1) {
            cout << "Number of classes must be greater than 1." << endl;
            return;
        }

        // Unknown/don't-care targets
        //TODO Handle infinite or nan numbers in the target and output.

        int numSamples = targets[0].size();

        _classes = numClasses;
        _samples = numSamples;
        //Transform outputs   (maximum value is set to 1 and other values to 0, column-wise)
        for (int col = 0; col < numSamples; col++) {
            double max = outputs[0][col];
            int ind = 0;

            for (int row = 1; row < numClasses; row++) {
                if (outputs[row][col] > max) {
                    max = outputs[row][col];
                    ind = row;
                }
                outputs[row][col] = 0.0;
            }
            outputs[0][col] = 0.0;
            outputs[ind][col] = 1;
        }

        //Confusion value
        int count = 0;
        for (int row = 0; row < numClasses; row++) {
            for (int col = 0; col < numSamples; col++) {
                if (targets[row][col] != outputs[row][col])
                    count++;
            }
        }
        double c = (double) count / (double) (2 * numSamples);

        // Confusion matrix
        vector<vector<int>> cm(numClasses, vector<int>(numClasses));
        for (int row = 0; row < numClasses; row++) {
            for (int col = 0; col < numClasses; col++) {
                cm[row][col] = 0;
            }
        }

        vector<int> i(numSamples);
        vector<int> j(numSamples);

        for (int col = 0; col < numSamples; col++) {
            for (int row = 0; row < numClasses; row++) {
                if (targets[row][col] == 1.0) {
                    i[col] = row;
                    break;
                }
            }
        }

        for (int col = 0; col < numSamples; col++) {
            for (int row = 0; row < numClasses; row++) {
                if (outputs[row][col] == 1.0) {
                    j[col] = row;
                    break;
                }
            }
        }

        for (int col = 0; col < numSamples; col++) {
            cm[i[col]][j[col]] = cm[i[col]][j[col]] + 1;
        }

        // Indices
        vector<vector<string>> ind(numClasses, vector<string>(numClasses));
        for (int row = 0; row < numClasses; row++)
            for (int col = 0; col < numClasses; col++)
                ind[row][col] = "";


        for (int col = 0; col < numSamples; col++) {
            if (ind[i[col]][j[col]] == "")
                ind[i[col]][j[col]] = to_string(col);
            else
                ind[i[col]][j[col]] =  ind[i[col]][j[col]].append(",").append(to_string(col));
        }

        // Percentages
        vector<vector<double>> per(numClasses, vector<double>(4));
        for (int row = 0; row < numClasses; row++) {
            for (int col = 0; col < 4; col++) {
                per[row][col] = 0.0;
            }
        }

        for (int row = 0; row < numClasses; row++) {
            vector<int> yi(numSamples);
            vector<int> ti(numSamples);
            for (int col = 0; col < numSamples; col++) {
                yi[col] = outputs[row][col];
                ti[col] = targets[row][col];

            }

            int a = 0, b = 0;
            for (int col = 0; col < numSamples; col++) {
                if (yi[col] != 1 && ti[col] == 1) a = a + 1;
                if (yi[col] != 1) b = b + 1;
            }
            per[row][0] = (double) a / (double) b;


            a = 0;
            b = 0;
            for (int col = 0; col < numSamples; col++) {
                if (yi[col] == 1 && ti[col] != 1) a = a + 1;
                if (yi[col] == 1) b = b + 1;
            }
            per[row][1] = (double) a / (double) b;


            a = 0;
            b = 0;
            for (int col = 0; col < numSamples; col++) {
                if (yi[col] == 1 && ti[col] == 1) a = a + 1;
                if (yi[col] == 1) b = b + 1;
            }
            per[row][2] = (double) a / (double) b;

            a = 0;
            b = 0;
            for (int col = 0; col < numSamples; col++) {
                if (yi[col] != 1 && ti[col] != 1) a = a + 1;
                if (yi[col] != 1) b = b + 1;
            }
            per[row][3] = (double) a / (double) b;

        }

        //NAN handling
        for (int row = 0; row < numClasses; row++) {
            for (int col = 0; col < 4; col++) {
                if (isnan(per[row][col]))
                    per[row][col] = 0.0;
            }
        }

        _c = c;
        _cm = cm;
        _ind = ind;
        _per = per;
    }

    string itos(int i) { // convert int to string
        stringstream s;
        s << i;
        return s.str();
    }

    float round(float valueToRound, int numberOfDecimalPlaces) {
        float multiplicationFactor = pow(10, numberOfDecimalPlaces);
        float interestedInZeroDPs = valueToRound * multiplicationFactor;
        return roundf(interestedInZeroDPs) / multiplicationFactor;
    }

    void printC() {
        cout << "\tConfusion value\n\t\tc = " << round(_c, 2) << endl;
    }

    void printCM() {
        cout << "\tConfusion Matrix" << endl;
        for (int row = 0; row < _classes; row++) {
            cout << "\t\t";
            for (int col = 0; col < _classes; col++) {
                cout << _cm[row][col] << " ";
            }
            cout << endl;
        }
    }

    void printInd() {
        cout << "\tIndices" << endl;
        for (int row = 0; row < _classes; row++) {
            for (int col = 0; col < _classes; col++) {
                cout << "\t\t[" << _ind[row][col] << "]";
            }
            cout << endl;
        }
    }

    void printPer() {
        cout << "\tPercentages" << endl;
        for (int row = 0; row < _classes; row++) {
            cout << "\t\t";
            for (int col = 0; col < 4; col++) {
                cout << round(_per[row][col], 2) << " ";
            }
            cout << endl;
        }
    }

    void print() {
        cout << "Confusion Results" << endl;
        cout << "=======================================" << endl;
        printC();
        printCM();
        printInd();
        printPer();
    }

    void print(vector<double> vec) {
        for (double d:vec) {
            cout << d << " ";
        }
        cout << endl;
    }

    void print(vector<vector<double>> vec) {
        for (unsigned i = 0; i < vec.size(); ++i) {
            for (unsigned j = 0; j < vec[0].size(); ++j) {
                cout << vec[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
};

#endif //ACCURACY_EVALUATION_CPP_CONFUSION_HPP
