/**
 * Author: Pranav Sukesh
 * Date of Creation: 1/26/2024
 * This project is an N-layer feed-forward neural network that can be both ran and trained using back propagation.
 * The user can specify specific configuration parameters using an external config file (read by the parser),
 * including the number of input, hidden, and output activation layers, the average error cutoff, the maximum number of
 * iterations, the lambda value, and the type of and range of weight population. The user can either let the network
 * randomize the initial weights, or they can specify a file for reading weights.
 * The network runs by calculating a weighted sum of the previous activation layer's activations and applying the
 * activation function to the sum, and the error is calculated as the squared difference between the expected and actual
 * values divided by 2. The truth table is reported to the user.
 * The network trains by utilizing back propagation to minimize error. It does so by first running the network, then
 * calculating the partial derivatives of the error with respect to each individual weight, then applying these changes
 * in weights to the weights. Through continued iterations of this process, the network either reaches the average error
 * cutoff or the maximum number of iterations. The average error, number of iterations, reason for terminating training,
 * and truth table is then reported to the user. If the user wishes to save the weights, the weights are saved to a
 * specified file.
 *
 * Table of Contents:
 * double sigmoid(double value)
 * double sigmoidDeriv(double value)
 * double tanh(double value)
 * double tanhDeriv(double value)
 * double linear(double value)
 * double linearDeriv(double value)
 * double randDouble()
 * void printTime(long milliseconds)
 * void setConfigurationParameters()
 * void echoConfigurationParameters()
 * void allocateMemory()
 * void randomizeWeights()
 * void saveWeights()
 * void loadWeights()
 * void populateArrays()
 * void runNetwork()
 * void runForTrain()
 * void trainNetwork()
 * void reportNetworkState()
 * void reportTruthTable()
 * void reportTrainingResults()
 * void reportResults()
 * void runAllTestCases()
 * int main(int argc, char* argv[])
 */

#include <iostream>
#include <chrono>
#include <cfloat>


#include "Parser.cpp"

using namespace std;


#define DIVIDER "------------------------------------------------------------------------------\n"

#define DEFAULT_CONFIG    "./ConfigFiles/defaultConfigFile.txt"

#define DEFAULT_PRECISION  12

#define MS_PER_SECOND      1000
#define SECONDS_PER_MINUTE 60
#define MINUTES_PER_HOUR   60
#define HOURS_PER_DAY      24
#define DAYS_PER_WEEK      7

#define IN_ACT_LAYER       0
#define OUT_ACT_LAYER      (numActLayers - 1)


typedef double* DARRAY1D;
typedef double** DARRAY2D;
typedef double*** DARRAY3D;

long timeSpentTraining;                         // time spent training the network in milliseconds

enum FUNCT_TYPE { SIGMOID, TANH, LINEAR };      // enum for types of activation functions

int numActLayers;
int* layerConfig;                               // array to store the number of activations in each activation layer

int precision;

FUNCT_TYPE actFunctName;
double (*actFunct)(double);                     // uses function pointers instead of conditionals to not branch paths
double (*actFunctDeriv)(double);

bool isTraining, isLoading, isSaving, isPrintingInput;
string configFileName, saveFilename, loadFilename, truthTableFilename;


double lambda, hiRange, lowRange;
int maxIterations, curIterations;
double totalError, avgError, avgErrCut;
int numTrainingSets, curTrainingCase;
int keepAliveInterval, savingInterval;

DARRAY2D a;                                     // all activations in the network

DARRAY3D weights;                               // all weights in the network

DARRAY2D runningOutputs;                        // stores the output of the network for all test cases
DARRAY2D truthTable;
DARRAY2D expectedValues;

DARRAY2D thetas;
DARRAY2D psis;

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
 * Evaluates the given value in the hyperbolic tangent function: y = (e^x - e^(-x)) / (e^x + e^(-x)).
 * @param value     the value to be evaluated in the hyperbolic tangent function
 * @return          the evaluated value
 */
double tanh(double value)
{
   double sign = value < 0.0 ? -1.0 : 1.0;
   double expVal = exp(-2.0 * sign * value);
   return sign * (1.0 - expVal) / (1.0 + expVal);
}

/**
 * Evaluates the given value in the hyperbolic tangent derivative: y = 1 - tanh(x)^2.
 * @param value     the value to be evaluated in the hyperbolic tangent derivative
 * @return          the evaluated value
 */
double tanhDeriv(double value)
{
   double tanhOutput = tanh(value);
   return 1.0 - tanhOutput * tanhOutput;
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
*    The number of layers and activations,
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

   numActLayers = parser::numActLayers;
   layerConfig = new int[numActLayers];
   parser::setLayerConfigData(layerConfig);

   isTraining = parser::isTraining;
   isPrintingInput = parser::isPrintingInput;

   precision = parser::precision > 0 ? parser::precision : DEFAULT_PRECISION;

   actFunctName = static_cast<FUNCT_TYPE>(parser::actFunctIndex);

   switch (actFunctName)
   {
      case SIGMOID:
         actFunct = sigmoid;
         actFunctDeriv = sigmoidDeriv;
         break;

      case TANH:
         actFunct = tanh;
         actFunctDeriv = tanhDeriv;
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
   curIterations = curTrainingCase = 0;

   return;
} // void setConfigurationParameters()

/**
* Presents the user with the configuration parameters for the network, including:
* The config file being read, the network configuration parameters, the running mode, the relevant files to
* read from/write to, and the runtime training parameters (if the network is training).
*/
void echoConfigurationParameters()
{
   int n;
   cout << DIVIDER;
   cout << "Reading Configuration Parameters from: " << configFileName <<
        (configFileName == DEFAULT_CONFIG ? " (Default)" : "") << endl;

   cout << DIVIDER;
   for (n = 0; n < OUT_ACT_LAYER; ++n)
   {
      cout << layerConfig[n] << "-";
   }
   cout << layerConfig[n] << " Network Configuration\n";
   cout << DIVIDER;

   cout << "Running Mode: " << (isTraining ? "Training" : "Running") << " Network\n";

   cout << "Activation Function: ";

   switch (actFunctName)
   {
      case SIGMOID:
         cout << "Sigmoid\n";
         break;

      case TANH:
         cout << "Hyperbolic Tangent\n";
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
   int j, n;

   a = new DARRAY1D[numActLayers];

   for (n = 0; n < numActLayers; ++n)
   {
      a[n] = new double[layerConfig[n]];
   }

   weights = new DARRAY2D[numActLayers];

   for (n = 1; n < numActLayers; ++n)
   {
      weights[n] = new DARRAY1D[layerConfig[n]];
      for (j = 0; j < layerConfig[n]; ++j)
      {
         weights[n][j] = new double[layerConfig[n - 1]];
      }
   }  // for (n = 0; n < numActLayers; ++n)

   truthTable = new DARRAY1D[numTrainingSets];
   expectedValues = new DARRAY1D[numTrainingSets];
   runningOutputs = new DARRAY1D[numTrainingSets];
   for (curTrainingCase = 0; curTrainingCase < numTrainingSets; ++curTrainingCase)
   {
      n = IN_ACT_LAYER;
      truthTable[curTrainingCase] = new double[layerConfig[n]];
      n = OUT_ACT_LAYER;
      expectedValues[curTrainingCase] = new double[layerConfig[n]];
      runningOutputs[curTrainingCase] = new double[layerConfig[n]];
   }  // for (curTrainingCase = 0; curTrainingCase < numTrainingSets; ++curTrainingCase)

   if (isTraining)
   {
      thetas = new DARRAY1D[numActLayers];
      psis = new DARRAY1D[numActLayers];

      for (n = 1; n < numActLayers; ++n)
      {
         thetas[n] = new double[layerConfig[n]];
         psis[n] = new double[layerConfig[n]];
      }

   } // if (isTraining)

   return;
} // void allocateMemory()

/**
* Randomizes the weights in each connectivity layer by using the randDouble() function to generate a random double
* between the specified high and low bounds.
*/
void randomizeWeights()
{
   int j, k, n;

   for (n = 1; n < numActLayers; ++n)
   {
      for (j = 0; j < layerConfig[n]; ++j)
      {
         for (k = 0; k < layerConfig[n - 1]; ++k)
         {
            weights[n][j][k] = randDouble();
         }
      } // for (j = 0; j < layerConfig[n]; ++j)
   } // for (n = 1; n < numActLayers; ++n)

   return;
} // void randomizeWeights()

/**
* Saves the weights to the specified output file, using the parser to write the weights to the specified file from the
* global arrays.
*/
void saveWeights()
{
   parser::saveWeightsToFile(weights, layerConfig);
   return;
} // void saveWeights()

/**
* Loads the weights for the network based on the specified input file, using the parser to read the weights from
* the file and store them in the global arrays.
*/
void loadWeights()
{
   parser::loadWeightsFromFile(weights, layerConfig);
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
* repeated for each subsequent activation layer. The theta values are stored locally.
*/
void runNetwork()
{
   int j, k, n;

   double theta;

   for (n = 1; n < numActLayers; ++n)
   {
      for (j = 0; j < layerConfig[n]; ++j)
      {
         theta = 0.0;
         for (k = 0; k < layerConfig[n - 1]; ++k)
         {
            theta += a[n - 1][k] * weights[n][j][k];      // weighted sum of the previous layer's activations
         }

         a[n][j] = actFunct(theta);
      } // for (j = 0; j < layerConfig[n]; ++j)
   } // for (n = 1; n < outActIndex; ++n)

   return;
} // void runNetwork()

/**
* Runs the network for the current test case and calculates psi values. This function is called while training the
* network, so it stores the theta and psi values in a global array.
*/
void runForTrain()
{
   int i, j, k, n;

   for (n = 1; n < OUT_ACT_LAYER; ++n)
   {
      for (j = 0; j < layerConfig[n]; ++j)
      {
         thetas[n][j] = 0.0;
         for (k = 0; k < layerConfig[n - 1]; ++k)
         {
            thetas[n][j] += a[n - 1][k] * weights[n][j][k];
         }

         a[n][j] = actFunct(thetas[n][j]);
      } // for (j = 0; j < layerConfig[n]; ++j)
   }  // for (n = 1; n < outActIndex; ++n)

   for (i = 0; i < layerConfig[n]; ++i)
   {
      thetas[n][i] = 0.0;
      for (j = 0; j < layerConfig[n - 1]; ++j)
      {
         thetas[n][i] += a[n - 1][j] * weights[n][i][j];
      }

      a[n][i] = actFunct(thetas[n][i]);
      psis[n][i] = (expectedValues[curTrainingCase][i] - a[n][i]) * actFunctDeriv(thetas[n][i]);
   } // for (i = 0; i < layerConfig[n]; ++i)

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
   int i, j, k, m, n;
   double omega;

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

      for (curTrainingCase = 0; curTrainingCase < numTrainingSets; ++curTrainingCase)
      {
         n = IN_ACT_LAYER;
         a[n] = truthTable[curTrainingCase];
         runForTrain();

         for (n = OUT_ACT_LAYER; n > 2; --n)
         {
            for (k = 0; k < layerConfig[n - 1]; ++k)
            {
               omega = 0.0;
               for (j = 0; j < layerConfig[n]; ++j)
               {
                  omega += psis[n][j] * weights[n][j][k];
                  weights[n][j][k] += lambda * psis[n][j] * a[n - 1][k];
               }

               psis[n - 1][k] = omega * actFunctDeriv(thetas[n - 1][k]);
            }  // for (k = 0; k < layerConfig[n]; ++k)
         }  // for (n = outActIndex; n > 2; --n)

         for (k = 0; k < layerConfig[n - 1]; ++k)
         {
            omega = 0.0;
            for (j = 0; j < layerConfig[n]; ++j)
            {
               omega += psis[n][j] * weights[n][j][k];
               weights[n][j][k] += lambda * psis[n][j] * a[n - 1][k];
            }

            psis[n - 1][k] = omega * actFunctDeriv(thetas[n - 1][k]);

            for (m = 0; m < layerConfig[n - 2]; ++m)
            {
               weights[n - 1][k][m] += lambda * psis[n - 1][k] * a[n - 2][m];
            }
         } // for (k = 0; k < layerConfig[n]; ++k)

         runNetwork();                                               // calculates new average error

         n = OUT_ACT_LAYER;
         for (i = 0; i < layerConfig[n]; ++i)
         {
            omega = (expectedValues[curTrainingCase][i] - a[n][i]);
            totalError += omega * omega * 0.5;
         }

      } // for (curTrainingCase = 0; curTrainingCase < numTrainingSets; ++curTrainingCase)

      avgError = totalError / static_cast<double>(numTrainingSets);

   } // while (avgError > avgErrCut && curIterations < maxIterations)

   endTime = chrono::high_resolution_clock::now();
   timeSpentTraining = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();

   return;
} // void trainNetwork()

/**
* Reports the current state of the network, including the inputs, expected output (if training), and actual output.
* This function is called after running the network for each test case.
*/
void reportNetworkState()
{
   int i, m, n;

   if (isPrintingInput)
   {
      cout << "[";
      cout << fixed << setprecision(precision) << truthTable[curTrainingCase][0];

      n = IN_ACT_LAYER;
      for (m = 1; m < layerConfig[n]; ++m)
      {
         cout << ", " << truthTable[curTrainingCase][m];
      }

      if (isTraining)
      {
         cout << "] | [";
         cout << fixed << setprecision(precision) << expectedValues[curTrainingCase][0];

         n = OUT_ACT_LAYER;
         for (i = 1; i < layerConfig[n]; ++i)
         {
            cout << ", " << expectedValues[curTrainingCase][i];
         }
      } // if (isTraining)

      cout << "] | [";

   }

   n = OUT_ACT_LAYER;
   cout << fixed << setprecision(precision) << runningOutputs[curTrainingCase][0];

   for (i = 1; i < layerConfig[n]; ++i)
   {
      cout << ", " << runningOutputs[curTrainingCase][i];
   }

   cout << (isPrintingInput ? "]" : "") << endl;

   return;
} // void reportNetworkState()

/**
* Reports the results of the network on all training sets based on the values provided by the most recent running.
* The user is presented with the truth table, expected output (if training), and actual output for each test case.
*/
void reportTruthTable()
{
   cout << DIVIDER;
   cout << "Truth Table: \n";

   if (isPrintingInput)
   {
      cout << "\nInput Activations";
      if (isTraining) cout << " | Expected Output";
      cout << " | ";
   }

   cout << "Actual Output\n";

   for (curTrainingCase = 0; curTrainingCase < numTrainingSets; ++curTrainingCase)
   {
      reportNetworkState();
   }

   cout << DIVIDER;

   return;
} // void reportTruthTable()

/**
* Reports the results of the network after training. The final average error and number of iterations are reported,
* along with the reason for exiting, time elapsed, and whether the weights were saved to a file.
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
* Reports the truth table, and actual output for each test case.
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
   int i, n;

   for (curTrainingCase = 0; curTrainingCase < numTrainingSets; ++curTrainingCase)
   {
      n = IN_ACT_LAYER;
      a[n] = truthTable[curTrainingCase];

      runNetwork();

      n = OUT_ACT_LAYER;
      for (i = 0; i < layerConfig[n]; ++i)
      {
         runningOutputs[curTrainingCase][i] = a[n][i];
      }

   }  // for (curTrainingCase = 0; curTrainingCase < numTrainingSets; ++curTrainingCase)

   return;
}  // void runAllTestCases()

/**
* Runs the neural network utilizing backpropagation, training or running based on the user's specifications.
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
   try
   {
      srand(time(NULL));                        // seeds the random number generator
      rand();

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

   }  // try
   catch (exception& e)
   {
      cerr << "Error: " << e.what() << endl;
   }  // catch (exception& e)

   return 0;
} // int main()