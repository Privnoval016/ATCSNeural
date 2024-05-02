/**
* Author: Pranav Sukesh
* Date of Creation: 1/26/2024
* This project is an A-B-C feed-forward neural network that can be both ran and trained using back propagation.
* The user can specify specific configuration parameters using an external config file (read by the parser),
* including the number of input, hidden, and output activation layers, the average error cutoff, the maximum number of
* iterations, the lambda value, and the type of and range of weight population. The user can either let the network
* randomize the initial weights, or they can specify a file for reading weights.
* The network runs by calculating a weighted sum of the previous activation layer's activations and applying the
* activation function to the sum, and the error is calculated as the squared difference between the expected and actual
* values divided by 2. The truth table is reported to the user.
* The network trains by utilizing gradient descent to minimize error. It does so by first running the network, then
* calculating the partial derivatives of the error with respect to each individual weight, then applying these changes
* in weights to the weights. Through continued iterations of this process, the network either reaches the average error
* cutoff or the maximum number of iterations. The average error, number of iterations, reason for terminating training,
* and truth table is then reported to the user. If the user wishes to save the weights, the weights are saved to a
* specified file.
*/

#include <iostream>
#include <chrono>
#include <cfloat>

#include "Parser.cpp"

using namespace std;


#define DIVIDER "------------------------------------------------------------------------------\n"

#define DEFAULT_CONFIG    "./ConfigFiles/defaultConfigFile.txt"

#define OUTPUT_PRECISION   17

#define MS_PER_SECOND      1000
#define SECONDS_PER_MINUTE 60
#define MINUTES_PER_HOUR   60
#define HOURS_PER_DAY      24
#define DAYS_PER_WEEK      7

typedef double* DARRAY1D;
typedef double** DARRAY2D;

long timeSpentTraining;                         // time spent training the network in milliseconds

enum FUNCT_TYPE { SIGMOID, LINEAR };            // enum for types of activation functions

int numInAct, numHidAct, numOutAct;

FUNCT_TYPE actFunctName;
double (*actFunct)(double);                     // uses function pointers instead of conditionals to not branch paths
double (*actFunctDeriv)(double);

bool isTraining, isLoading, isSaving;
string configFileName, saveFilename, loadFilename, truthTableFilename;


double lambda, hiRange, lowRange;
int maxIterations, curIterations;
double totalError, avgError, avgErrCut;
int numTrainingSets, curTestCase;
int keepAliveInterval, savingInterval;

DARRAY1D a;                                     // input activations
DARRAY1D h;                                     // hidden activations
DARRAY1D F;                                     // output activation

DARRAY2D kjWeights;                             // weights between the hidden and input activations
DARRAY2D jiWeights;                             // weights between the hidden and output activations

DARRAY2D truthTable;
DARRAY2D expectedValues;

DARRAY1D jThetas;

DARRAY1D iPsis;

/**
* Evaluates the given value in the sigmoid function: y = 1 / (1 + e^(-x)).
* @param value   the value to be evaluated in the sigmoid function
* @return        the evaluated value
*/
double sigmoid(double value)
{
   return 1.0 / (1.0 + exp(-value));
}

/**
* Evaluates the given value in the sigmoid derivative: y = sigmoid(x) * (1 - sigmoid(x)).
* @param value   the value to be evaluated in the sigmoid derivative
* @return        the evaluated value
*/
double sigmoidDeriv(double value)
{
   double sigmoidOutput = sigmoid(value);
   return sigmoidOutput * (1.0 - sigmoidOutput);
}

/**
* Evaluates the given value in the linear function: y = x.
* @param value   the value to be evaluated in the linear function
* @return        the evaluated value
*/
double linear(double value)
{
   return value;
}

/**
* Evaluates the given value in the linear function's derivative: y = 1.
* @param value   the value to be evaluated in the linear function's derivative
* @return        the evaluated value
*/
double linearDeriv(double value)
{
   return 1.0;
}

/**
* Generates a random double between the specified high and low bounds (outlined in the global variables
* when setting configuration parameters).
* @return  a random double between the specified high and low bounds.
*/
double randDouble()
{
   return lowRange + (static_cast<double>(rand()) / RAND_MAX) * (hiRange - lowRange);
}

/**
* Accepts a time in milliseconds and prints the time in milliseconds, seconds, minutes, hours, days, and weeks.
* @param milliseconds the time in milliseconds
*/
void printTime(long milliseconds)
{
   long seconds, minutes, hours, days, weeks;

   cout << "Elapsed time: ";

   if (milliseconds > MS_PER_SECOND)
   {
      seconds = milliseconds / MS_PER_SECOND;
      milliseconds %= MS_PER_SECOND;

      if (seconds > SECONDS_PER_MINUTE)
      {
         minutes = seconds / SECONDS_PER_MINUTE;
         seconds %= SECONDS_PER_MINUTE;

         if (minutes > MINUTES_PER_HOUR)
         {
            hours = minutes / MINUTES_PER_HOUR;
            minutes %= MINUTES_PER_HOUR;

            if (hours > HOURS_PER_DAY)
            {
               days = hours / HOURS_PER_DAY;
               hours %= HOURS_PER_DAY;

               if (days > DAYS_PER_WEEK)
               {
                  weeks = days / DAYS_PER_WEEK;
                  days %= DAYS_PER_WEEK;

                  cout << weeks << " weeks, ";
               } // if (days > DAYS_PER_WEEK)

               cout << days << " days, ";
            } // if (hours > HOURS_PER_DAY)

            cout << hours << " hours, ";
         } // if (minutes > MINUTES_PER_HOUR)

         cout << minutes << " minutes, ";
      } // if (seconds > SECONDS_PER_MINUTE)

      cout << seconds << " seconds, ";
   } // if (milliseconds > MS_PER_SECOND)

   cout << milliseconds << " milliseconds\n\n";

   return;
} // void printTime(long milliseconds)

/**
* Sets the configuration parameters for the network by reading them from the config file using the parser.
* These parameters include:
*    The number of input, hidden, and output activations,
*    Whether the network is training or running,
*    The relevant file names for loading and saving weights, the truth table, and the config file,
*    The activation function used in the network,
*    The keep alive/saving intervals,
*    The type of weight population,
*    The range of the random weights,
*    The lambda value applied in training,
*    The error cutoff for training (and its accumulator),
*    The maximum allowed iterations when training (and its accumulator),
*    And the number of specified training sets.
*/
void setConfigurationParameters()
{
   parser::readConfigFile(configFileName);

   numInAct = parser::numInAct;
   numHidAct = parser::numHidAct;
   numOutAct = parser::numOutAct;

   isTraining = parser::isTraining;

   actFunctName = static_cast<FUNCT_TYPE>(parser::actFunctIndex);

   switch (actFunctName)
   {
      case SIGMOID:
         actFunct = sigmoid;
      actFunctDeriv = sigmoidDeriv;
      break;

      case LINEAR:
         actFunct = linear;
      actFunctDeriv = linearDeriv;
      break;

   } // switch (actFunctName)

   isLoading = parser::isLoading;
   isSaving = parser::isSaving;

   keepAliveInterval = parser::keepAliveInterval;

   savingInterval = parser::savingInterval;

   loadFilename = parser::loadFilename;
   saveFilename = parser::saveFilename;

   truthTableFilename = parser::truthTableFile;

   hiRange = parser::hiRange;
   lowRange = parser::lowRange;

   lambda = parser::lambda;
   avgErrCut = parser::avgErrCut;
   totalError = avgError = 0.0;

   maxIterations = parser::maxIterations;
   numTrainingSets = parser::numTrainingSets;
   curIterations = curTestCase = 0;

   return;
} // void setConfigurationParameters()

/**
* Presents the user with the configuration parameters for the network, including:
* The config file being read, the network configuration parameters, the running mode, the relevant files to
* read from/write to, and the runtime training parameters (if the network is training).
*/
void echoConfigurationParameters()
{
   cout << DIVIDER;
   cout << "Reading Configuration Parameters from: " << configFileName <<
      (configFileName == DEFAULT_CONFIG ? " (Default)" : "") << endl;

   cout << DIVIDER;
   cout << numInAct << "-" << numHidAct << "-" << numOutAct << " Network Configuration Parameters\n";
   cout << DIVIDER;
   cout << "Running Mode: " << (isTraining ? "Training" : "Running") << " Network\n";
   cout << "Activation Function: ";

   switch (actFunctName)
   {
      case SIGMOID:
         cout << "Sigmoid\n";
         break;

      case LINEAR:
         cout << "Linear\n";
         break;

   } // switch (actFunctName)

   cout << DIVIDER;

   if (isLoading)
      cout << "Loading weights from file: " << loadFilename << endl;
   else
      cout << "Randomizing weights between [" << lowRange << ", " << hiRange << "]\n";

   if (isSaving)
      cout << "Saving weights to file: " << saveFilename << endl;
   else
      cout << "Not saving weights to file\n";

   cout << "Truth Table obtained from: " << truthTableFilename << " (" << numTrainingSets << " test cases)\n";

   if (isTraining)
   {
      cout << "\nRuntime Training Parameters:\n";
      cout << "Random range for weights: [" << lowRange << ", " << hiRange << "]\n";
      cout << "Max iterations:           " << maxIterations << endl;
      cout << "Error threshold:          " << avgErrCut << endl;
      cout << "Lambda Value:             " << lambda << endl;

      if (keepAliveInterval > 0)
         cout << "Keep Alive Interval:      " << keepAliveInterval << " iterations\n";
      else
         cout << "Keep Alive Interval:      Disabled\n";

      if (isSaving)
      {
         if (savingInterval > 0)
            cout << "Saving interval:          " << savingInterval << " iterations\n";
         else
            cout << "Saving interval:          Disabled\n";
      }
      cout << DIVIDER;
   }  // if (isTraining)

   return;
} // void echoConfigurationParameters()

/**
* Allocates memory for the arrays used in the network, including:
* The input/hidden/output activation arrays,
* The weights between the hidden and input activations and the hidden and output activations,
* The truth table and expected values for the training sets,
* And training-specific arrays (if the network is training), including:
*    The thetas (a weighted sum of the previous layer's activations),
*    The psi values (omega * theta evaluated in the derivative of the activation function),
*/
void allocateMemory()
{
   int i, j;

   a = new double[numInAct];
   h = new double[numHidAct];
   F = new double[numOutAct];

   kjWeights = new DARRAY1D[numHidAct];
   for (j = 0; j < numHidAct; ++j) kjWeights[j] = new double[numInAct];
   jiWeights = new DARRAY1D[numOutAct];
   for (i = 0; i < numOutAct; ++i) jiWeights[i] = new double[numHidAct];

   truthTable = new DARRAY1D[numTrainingSets];
   expectedValues = new DARRAY1D[numTrainingSets];
   for (curTestCase = 0; curTestCase < numTrainingSets; ++curTestCase)
   {
      truthTable[curTestCase] = new double[numInAct];
      expectedValues[curTestCase] = new double[numOutAct];
   }

   if (isTraining)
   {
      jThetas = new double[numHidAct];
      iPsis = new double[numOutAct];

   } // if (isTraining)

   return;
} // void allocateMemory()

/**
* Randomizes the weights between the hidden and input activations and the hidden and output activations by using the
* randDouble() function to generate a random double between the specified high and low bounds.
*/
void randomizeWeights()
{
   int i, j, k;

   for (j = 0; j < numHidAct; ++j)
   {
      for (k = 0; k < numInAct; ++k)
      {
         kjWeights[j][k] = randDouble();
      }
   } // for (j = 0; j < numHidAct; ++j)

   for (i = 0; i < numOutAct; ++i)
   {
      for (j = 0; j < numHidAct; ++j)
      {
         jiWeights[i][j] = randDouble();
      }
   } // for (i = 0; i < numOutAct; ++i)

   return;
} // void randomizeWeights()

/**
* Saves the weights for an A-B-C network to the specified output file, using the parser to write the weights to the
* specified file from the global arrays.
*/
void saveWeights()
{
   parser::saveWeightsToFile(kjWeights, jiWeights);
   return;
} // void saveWeights()

/**
* Loads the weights for the network based on the specified input file, using the parser to read the weights from
* the file and store them in the global arrays.
*/
void loadWeights()
{
   parser::loadWeightsFromFile(kjWeights, jiWeights);
   return;
} // void loadWeights()

/**
* Populates the arrays utilized in the network, that were created in the allocateMemory() function. The truth table is
* populated based on the specified file in the config file, and the expected values are populated based on the truth
* table. If the network is loading weights, the weights are loaded from the specified file. Otherwise, the weights are
* randomized based on the specified range in the config file.
*/
void populateArrays()
{
   parser::loadTruthTableFromFile(truthTable, expectedValues);

   if (isLoading)                         // Loads the weights from a specified file
      loadWeights();
   else                                   // Randomizes the weights
      randomizeWeights();

   return;
} // void populateArrays()

/**
* Runs the network for the current test case. The network calculates the weighted sum of the input activations and
* applies the activation function to the sum to determine the values for each hidden activation. This process is
* repeated for the output activation. The theta and omega values are stored locally.
*/
void runNetwork()
{
   int i, j, k;

   double iTheta, jTheta;

   for (j = 0; j < numHidAct; ++j)
   {
      jTheta = 0.0;
      for (k = 0; k < numInAct; ++k)
      {
         jTheta += a[k] * kjWeights[j][k];      // weighted sum of the input activations
      }
      h[j] = actFunct(jTheta);
   } // for (j = 0; j < numHidAct; ++j)

   for (i = 0; i < numOutAct; ++i)
   {
      iTheta = 0.0;
      for (j = 0; j < numHidAct; ++j)
      {
         iTheta += h[j] * jiWeights[i][j];      // weighted sum of the hidden activations
      }
      F[i] = actFunct(iTheta);
   } // for (i = 0; i < numOutAct; ++i)

   return;
} // void runNetwork()

/**
* Runs the network for the current test case and calculates the error. This function is called while training the
* network, so it stores the theta and psi values in a global array.
*/
void runForTrain()
{
   int i, j, k;
   double iOmega, iTheta;

   for (j = 0; j < numHidAct; ++j)
   {
      jThetas[j] = 0.0;

      for (k = 0; k < numInAct; ++k)
      {
         jThetas[j] += a[k] * kjWeights[j][k];      // weighted sum of the input activations
      }

      h[j] = actFunct(jThetas[j]);
   } // for (j = 0; j < numHidAct; ++j)

   for (i = 0; i < numOutAct; ++i)
   {
      iTheta = 0.0;

      for (j = 0; j < numHidAct; ++j)
      {
         iTheta += h[j] * jiWeights[i][j];      // weighted sum of the hidden activations
      }

      F[i] = actFunct(iTheta);
      iOmega = expectedValues[curTestCase][i] - F[i];
      iPsis[i] = iOmega * actFunctDeriv(iTheta);

   } // for (i = 0; i < numOutAct; ++i)

   return;
} // void runForTrain()

/**
* Trains the neural network to minimize error through gradient descent, utilizing the backpropagation algorithm to
* optimize the efficiency. The network trains by first running the network for each test case, then calculating the
* partial derivatives of the error with respect to each individual weight, then applying these changes in weights to the
* weights, repeating this process until the average error reaches the cutoff or the maximum number of iterations is
* achieved. If specified, the weights are saved to a file at a specified interval.
*/
void trainNetwork()
{
   int i, j, k;
   double iOmega, jOmega, jPsi;

   avgError = DBL_MAX;
   auto startTime = chrono::high_resolution_clock::now();
   auto endTime = chrono::high_resolution_clock::now();

   while (avgError > avgErrCut && curIterations < maxIterations)           // trains the network
   {
      curIterations++;
      totalError = 0.0;

      if (keepAliveInterval > 0 && curIterations % keepAliveInterval == 0)
      {
         cout << "Iteration " << curIterations << ": Average Error: " << avgError << endl;

         endTime = chrono::high_resolution_clock::now();
         timeSpentTraining = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
         printTime(timeSpentTraining);
      }  // if (keepAliveInterval > 0 && curIterations % keepAliveInterval == 0)

      if (isSaving && savingInterval > 0 && curIterations % savingInterval == 0)
      {
         cout << "Iteration " << curIterations << ": Saving weights to file: " << saveFilename << endl << endl;
         saveWeights();
      }

      for (curTestCase = 0; curTestCase < numTrainingSets; ++curTestCase)
      {
         a = truthTable[curTestCase];
         runForTrain();                                     // calculates activations, thetas and psis

         for (j = 0; j < numHidAct; ++j)                    // calculates changes in weights
         {
            jOmega = 0.0;
            for (i = 0; i < numOutAct; ++i)
            {
               jOmega += iPsis[i] * jiWeights[i][j];
               jiWeights[i][j] += lambda * iPsis[i] * h[j];
            }

            jPsi = jOmega * actFunctDeriv(jThetas[j]);

            for (k = 0; k < numInAct; ++k)
            {
               kjWeights[j][k] += lambda * jPsi * a[k];
            }
         } // for (j = 0; j < numHidAct; ++j)

         runNetwork();                                      // calculates new average error

         for (i = 0; i < numOutAct; ++i)
         {
            iOmega = expectedValues[curTestCase][i] - F[i];
            totalError += iOmega * iOmega * 0.5;
         }

      } // for (curTestCase = 0; curTestCase < numTrainingSets; ++curTestCase)

      avgError = totalError / static_cast<double>(numTrainingSets);

   } // while (avgError > avgErrCut && curIterations < maxIterations)

   endTime = chrono::high_resolution_clock::now();
   timeSpentTraining = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();

   return;
} // void trainNetwork()

/**
* Reports the current state of the network, including the input activations, expected output, and actual output.
* This function is called after running the network for each test case.
*/
void reportNetworkState()
{
   int i, k;

   cout << "Input:           [";
   cout << fixed << setprecision(OUTPUT_PRECISION) << truthTable[curTestCase][0];

   for (k = 1; k < numInAct; ++k)
   {
      cout << ", " << truthTable[curTestCase][k];
   }

   if (isTraining)
   {
      cout << "]\nExpected Output: [";
      cout << expectedValues[curTestCase][0];

      for (i = 1; i < numOutAct; ++i)
      {
         cout << ", " << expectedValues[curTestCase][i];
      }
   } // if (isTraining)

   cout << "]\nActual Output:   [";
   cout << F[0];

   for (i = 1; i < numOutAct; ++i)
   {
      cout << ", " << F[i];
   }

   cout << "] \n";

   return;
} // void reportNetworkState()

/**
* Reports the results of the network on all training sets after running on the truth table with the current weights.
* The user is presented with the truth table, expected output, and actual output for each test case.
*/
void reportTruthTable()
{
   int i, k;

   cout << DIVIDER;
   cout << "Truth Table: \n";

   for (curTestCase = 0; curTestCase < numTrainingSets; ++curTestCase)
   {
      cout << "\nCase " << (curTestCase + 1) << ":\n";

      a = truthTable[curTestCase];
      runNetwork();
      reportNetworkState();
   } // for (curTestCase = 0; curTestCase < numTrainingSets; ++curTestCase)

   cout << DIVIDER;

   return;
} // void reportTruthTable()

/**
* Reports the results of the network after training. The final average error and number of iterations are reported,
* along with the reason for exiting, time elapsed, truth table, expected output, and actual output for each test case
* when running the network with the optimized weights. If the user wishes to save the weights, the weights are saved
* to the specified file.
*/
void reportTrainingResults()
{
   cout << DIVIDER;
   cout << "Training Results:\n";
   cout << DIVIDER;

   cout << "Final Average Error:  " << avgError << endl;
   cout << "Number of Iterations: " << curIterations << endl << endl;

   printTime(timeSpentTraining);

   cout << "Ended training after reaching the: \n";
   if (curIterations >= maxIterations)
      cout << "Maximum Number of Iterations\n";
   if (avgError <= avgErrCut)
      cout << "Average Error Cutoff\n";

   if (isSaving)
   {
      cout << "\nSaving weights to file: " << saveFilename << endl;
   }

   return;
} // void reportTrainingResults()

/**
* Reports the results of the network.
* If the network is training:
* The final average error and number of iterations are reported, along with the truth table, expected output,
* and actual output. If the user wishes to save the weights, the weights are saved to the specified file.
* If the network is running:
* Reports the truth table, expected output, and actual output for each test case.
*/
void reportResults()
{
   if (isTraining) reportTrainingResults();

   reportTruthTable();

   return;
} // void reportResults()

/**
* Runs the network for all test cases in the truth table and updates the stored activations.
*/
void runAllTestCases()
{
   for (curTestCase = 0; curTestCase < numTrainingSets; ++curTestCase)
   {
      a = truthTable[curTestCase];
      runNetwork();
   }

   return;
}  // void runAllTestCases()

/**
* Runs the A-B-C neural network utilizing backpropagation, training or running based on the user's specifications.
* The network first sets the configuration parameters and presents them to the user, then allocates memory for the
* arrays used in either training or running the network. The arrays are then populated with their required values, and
* the network is either trained or run. The results are then reported to the user, and if training, the weights may be
* saved to a file.
*
* @param argc the number of command line arguments
* @param argv the command line arguments
* @return     0 if the program runs successfully
*/
int main(int argc, char* argv[])
{
   srand(time(NULL));

   configFileName = (argc > 1) ? argv[1] : DEFAULT_CONFIG;

   setConfigurationParameters();
   echoConfigurationParameters();

   allocateMemory();
   populateArrays();

   if (isTraining)
      trainNetwork();

   runAllTestCases();

   if (isSaving)
      saveWeights();

   reportResults();

   return 0;
} // int main()