#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <iomanip>
#include <math.h>
#include <fstream>

#include "read_csv.cuh"
#include "cuda_stuff.cuh" // for matrix indexing

using namespace std;

/////////////////////////////////////////////////////////
// Functions for reading the dataset from a file
/////////////////////////////////////////////////////////

/* Read a csv file with a given number of rows and columns */
void read_csv(const char *filename, float *data_array, int nbrow, int nbcol, bool verbose /*= false */)
{
    string row_as_string;
    string value;
    double ioTemp;
    ifstream infile;
    infile.open(filename, ifstream::in);
    int row_count = 0;
    if (infile.is_open())
    {
        // read the headers (and discard)
        getline(infile, row_as_string, '\n');
        if (verbose)
            cout << "headers: " << row_as_string << "!" << std::endl;
        for (int i = 0; i < nbrow; i++)
        {
            getline(infile, row_as_string, '\n');
            // cout << "read line " << row_as_string << "!" << std::endl;
            istringstream line_stream(row_as_string);
            for (int j = 0; j < nbcol; j++)
            {
                getline(line_stream, value, ',');
                ioTemp = strtod(value.c_str(), NULL);
                // cout << "("<<i<<","<<j<<") = "<< ioTemp << std::endl;

                data_array[IDX2C(i, j, nbrow)] = ioTemp;
            }
            ++row_count;
        }
        infile.close();
        if (verbose)
            cout << "Read " << row_count << " rows." << std::endl;
    }
    else
        cout << "Cannot open file." << endl;
}
