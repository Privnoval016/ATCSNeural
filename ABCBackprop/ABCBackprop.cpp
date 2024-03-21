/**
 * Author: Pranav Sukesh
 * Date of Creation: 1/26/2024
 * This project is an A-B-C feed-forward neural network that can be both trained and ran.
 * The user can specify specific configuration parameters for the network in the setConfigurationParameters() function,
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

#include "fileparser.h"
#include <cfloat>

using namespace std;


#define DIVIDER "------------------------------------------------------------------------------\n"

#define DEFAULT_CONFIG    "../ConfigFiles/configfile.txt"

#define OUTPUT_PRECISION   17

#define MS_PER_SECOND      1000
#define SECONDS_PER_MINUTE 60
#define MINUTES_PER_HOUR   60
#define HOURS_PER_DAY      24
#define DAYS_PER_WEEK      7

typedef double* DARRAY1D;
typedef double** DARRAY2D;

time_t startTime, endTime;

enum FUNCT_TYPE { SIGMOID, LINEAR };             // enum for types of activation functions

int numInAct, numHidAct, numOutAct;

bool isTraining;

FUNCT_TYPE actFunctName;
double (*actFunct)(double);                     // uses function pointers instead of conditionals to not branch paths
double (*actFunctDeriv)(double);

bool isLoading, isSaving;
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
 * \brief         Evaluates the given value in the sigmoid function: y = 1 / (1 + e^(-x)).
 * \param value   the value to be evaluated in the sigmoid function
 * \return        the evaluated value
 */
double sigmoid(double value)
{
   return 1.0 / (1.0 + exp(-value));
}

/**
 * \brief         Evaluates the given value in the sigmoid derivative: y = sigmoid(x) * (1 - sigmoid(x)).
 * \param value   the value to be evaluated in the sigmoid derivative
 * \return        the evaluated value
 */
double sigmoidDeriv(double value)
{
   double sigmoidOutput = sigmoid(value);
   return sigmoidOutput * (1.0 - sigmoidOutput);
}

/**
 * \brief         Evaluates the given value in the linear function: y = x.
 * \param value   the value to be evaluated in the linear function
 * \return        the evaluated value
 */
double linear(double value)
{
   return value;
}

/**
 * \brief         Evaluates the given value in the linear function's derivative: y = 1.
 * \param value   the value to be evaluated in the linear function's derivative
 * \return        the evaluated value
 */
double linearDeriv(double value)
{
   return 1.0;
}

/**
 * \brief   Generates a random double between the specified high and low bounds (outlined in the global variables
 *          when setting configuration parameters).
 * \return  a random double between the specified high and low bounds.
 */
double randDouble()
{
   return lowRange + (static_cast<double>(rand()) / RAND_MAX) * (hiRange - lowRange);
}

/*
 * Accept a value representing seconds elapsed and print out a decimal value in easier to digest units.
 * @param seconds the number of seconds elapsed
 */
void printTime(double seconds)
{
   double minutes, hours, days, weeks;

   cout << "Elapsed time: ";

   if (seconds < 1.)
      cout << (seconds * MS_PER_SECOND) << " milliseconds";
   else if (seconds < SECONDS_PER_MINUTE)
      cout << seconds << " seconds";
   else
   {
      minutes = seconds / SECONDS_PER_MINUTE;

      if (minutes < MINUTES_PER_HOUR)
         cout << minutes << " minutes";
      else
      {
         hours = minutes / MINUTES_PER_HOUR;

         if (hours < HOURS_PER_DAY)
            cout << hours << " hours";
         else
         {
            days = hours / HOURS_PER_DAY;

            if (days < DAYS_PER_WEEK)
               cout << days << " days";
            else
            {
               weeks = days / DAYS_PER_WEEK;

               cout << weeks << " weeks";
            }
         } // if (hours < HOURS_PER_DAY)...else
      } // if (minutes < MINUTES_PER_HOUR)...else
   } // else if (seconds < SECONDS_PER_MINUTE)...else

   printf("\n\n");
   return;
} // void printTime(double seconds)


/**
 * \brief   Sets the configuration parameters for the network. These parameters include:
 *          The number of input, hidden, and output activations,
 *          Whether the network is training or running,
 *          The type of weight population (and information for saving and loading weights),
 *          The range of the random weights,
 *          The lambda value applied in training,
 *          The error cutoff for training (and its accumulator),
 *          The maximum allowed iterations when training (and its accumulator),
 *          And the number of specified training sets,
 */
void setConfigurationParameters()
{
   parser::readConfigFile(configFileName);

   numInAct = stoi(parser::getParameter("NUM_IN_ACT"));
   numHidAct = stoi(parser::getParameter("NUM_HID_ACT"));
   numOutAct = stoi(parser::getParameter("NUM_OUT_ACT"));

   if (stoi(parser::getParameter("RUN_MODE")) == 0)
      isTraining = false;
   else
      isTraining = true;

   actFunctName = static_cast<FUNCT_TYPE>(stoi(parser::getParameter("ACT_FUNCT")));

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


   if (stoi(parser::getParameter("LOAD_WEIGHTS")) == 0)
      isLoading = false;
   else
      isLoading = true;

   if (stoi(parser::getParameter("SAVE_WEIGHTS")) == 0)
      isSaving = false;
   else
      isSaving = true;

   keepAliveInterval = stoi(parser::getParameter("KEEP_ALIVE"));

   if (isSaving) savingInterval = stoi(parser::getParameter("SAVE_INTERVAL"));

   if (isLoading) loadFilename = parser::getParameter("LOAD_FILE");
   if (isSaving) saveFilename = parser::getParameter("SAVE_FILE");

   truthTableFilename = parser::getParameter("TRUTH_TABLE_FILE");

   if (isTraining)
   {
      hiRange = stod(parser::getParameter("HI_RANGE"));
      lowRange = stod(parser::getParameter("LOW_RANGE"));

      lambda = stod(parser::getParameter("LAMBDA"));
      avgErrCut = stod(parser::getParameter("MIN_ERROR"));
      totalError = avgError = 0.0;

      maxIterations = stoi(parser::getParameter("MAX_ITER"));
      numTrainingSets = stoi(parser::getParameter("NUM_TEST_CASES"));
      curIterations = curTestCase = 0;
   }

   return;
} // void setConfigurationParameters()

/**
 * \brief   Presents the user with the current configuration parameters for the network, set in the
 *          setConfigurationParameters() function, to ensure that the user is aware of the current
 *          specifications of the network.
 */
void echoConfigurationParameters()
{
   cout << DIVIDER;
   cout << "Reading Configuration Parameters from: " << "../configfile.txt\n";
   cout << DIVIDER;
   cout << numInAct << "-" << numHidAct << "-" << numOutAct << " Network Configuration Parameters\n";
   cout << DIVIDER;
   cout << "Running Mode: " << (isTraining ? "Training" : "Running") << " Network\n";
   cout << DIVIDER;

   if (isLoading)
      cout << "Loading weights from file: " << loadFilename << endl;
   else
      cout << "Randomizing weights between [" << lowRange << ", " << hiRange << "]\n";

   if (isSaving)
      cout << "Saving weights to file: " << saveFilename << endl;

   cout << "Truth Table obtained from: " << truthTableFilename << " (" << numTrainingSets << " test cases)\n";

   if (isTraining)
   {
      cout << "\nRuntime Training Parameters:\n";
      cout << "Random range for weights: [" << lowRange << ", " << hiRange << "]\n";
      cout << "Max iterations:           " << maxIterations << endl;
      cout << "Error threshold:          " << avgErrCut << endl;
      cout << "Lambda Value:             " << lambda << endl;
      cout << "Saving weights every " << savingInterval << " iterations\n";
      cout << DIVIDER;
   }
   return;
} // void echoConfigurationParameters()

/**
 * \brief   Allocates memory for the arrays used in the network, including:
 *          The input/hidden/output activation arrays,
 *          The weights between the hidden and input activations and the hidden and output activations,
 *          The truth table and expected values for the training sets,
 *          And training-specific arrays (if the network is training), including:
 *             The thetas (a weighted sum of the previous layer's activations),
 *             The omega values (the difference between the expected and actual output),
 *             The psi values (omega * theta evaluated in the derivative of the activation function),
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
 * \brief  Randomizes the weights between the hidden and input activations and the hidden and output activations by
 *         using the randDouble() function to generate a random double between the specified high and low bounds.
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
 * \brief Saves the weights for an A-B-C network based on the specified output file.
 *        The format of the file is as follows: the first line contains the configuration of the network (number of
 *        input, hidden, and output activations), and the following lines contain the weights between each hidden
 *        activation and the input activations, followed by the weights between each output activation and the hidden
 *        activations (each new activation node is separated by a blank line). The saved weights are the weights
 *        that the network produces after training.
 *
 */
void saveWeights()
{
   parser::saveWeightsToFile(kjWeights, jiWeights);
   return;
} // void saveWeights()

/**
 *
 */
void loadWeights()
{
   parser::loadWeightsFromFile(kjWeights, jiWeights);
   return;
} // void loadWeights()

/**
 * \brief   Populates the arrays utilized in the network, that were created in the allocateMemory() function.
 *          The truth table is populated with combinations of 0 and 1, and the expected value is changed by the user to
 *          run different bitwise operators. Additionally, the weights can be populated with either randomized values
 *          or loaded based on a specified file.
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
 * \brief   Runs the network for the current test case. The network calculates the weighted sum of the input activations
 *          and applies the activation function to the sum to determine the values for each hidden activation. This
 *          process is repeated for the output activation. The theta and omega values are stored locally.
 */
void runNetwork()
{
   int i, j, k;

   double iTheta, jTheta;

   a = truthTable[curTestCase];

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
 * \brief   Runs the network for the current test case and calculates the error. This function is called while training
 *          the network, so it stores the theta values in a global array.
 */
void runForTrain()
{
   int i, j, k;
   double iOmega, iTheta;

   a = truthTable[curTestCase];
   totalError = 0.0;

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
      totalError += iOmega * iOmega * 0.5;
   } // for (i = 0; i < numOutAct; ++i)

   return;
} // void runForTrain()

/**
 * \brief   Trains the neural network to minimize error through gradient descent. The network runs forward for
 *          each test case, calculating the error. The partial derivatives of the error with respect to each individual
 *          weight between the hidden and output activation layers are calculated, then are multiplied by lambda to
 *          determine the change in weights for the next iteration. A similar process is done for the weights between
 *          the input and hidden activation layers, and the weights are updated accordingly. The process continues
 *          until either the average error is less than the error cutoff or the maximum number of iterations is reached.
 */
void trainNetwork()
{
   int i, j, k;
   double iOmega, jOmega, jPsi;

   avgError = DBL_MAX;

   while (avgError > avgErrCut && curIterations < maxIterations)           // trains the network
   {
      curIterations++;
      totalError = 0.0;

      if (curIterations % keepAliveInterval == 0)
      {
         cout << "Iteration " << curIterations << ": Average Error: " << avgError << endl;
         time(&endTime);
         printTime(difftime(endTime, startTime));
      }

      if (isSaving && curIterations % savingInterval == 0)
      {
         cout << "Iteration " << curIterations << ": Saving weights to file: " << saveFilename << endl;
         saveWeights();
      }

      for (curTestCase = 0; curTestCase < numTrainingSets; ++curTestCase)
      {
         runForTrain();

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

      avgError = totalError / static_cast<float>(numTrainingSets);

   } // while (avgError > avgErrCut && curIterations < maxIterations)

   return;
} // void trainNetwork()

/**
 * \brief   Reports the current state of the network, including the input activations, expected output, and actual output.
 *          This function is called after running the network for each test case.
 */
void reportNetworkState()
{
   int i, k;

   cout << "[";
   cout << fixed << setprecision(OUTPUT_PRECISION) << truthTable[curTestCase][0];

   for (k = 1; k < numInAct; ++k)
   {
      cout << ", " << truthTable[curTestCase][k];
   }

   cout << "] | [";
   cout << expectedValues[curTestCase][0];

   for (i = 1; i < numOutAct; ++i)
   {
      cout << ", " << expectedValues[curTestCase][i];
   }

   cout << "] | [";
   cout << F[0];

   for (i = 1; i < numOutAct; ++i)
   {
      cout << ", " << F[i];
   }

   cout << "] \n";

   return;
} // void reportNetworkState()

/**
 * \brief   Reports the results of the network after running on the truth table with the current weights. The user is
 *          presented with the truth table, expected output, actual output, and error for each test case.
 */
void reportTruthTable()
{
   int i, k;

   cout << DIVIDER;
   cout << "Truth Table: \n";

   cout << "Input | Expected Output | Actual Output\n";

   for (curTestCase = 0; curTestCase < numTrainingSets; ++curTestCase)
   {
      cout << "Case " << (curTestCase + 1) << ": ";

      runNetwork();
      reportNetworkState();

   } // for (curTestCase = 0; curTestCase < numTrainingSets; ++curTestCase)

   cout << DIVIDER;

   return;
} // void reportTruthTable()

/**
 * \brief   Reports the results of the network after training.
 *          The final average error and number of iterations are reported, along with the truth table, expected output,
 *          actual output, and error for each test case when running the network with the optimized weights. If the user
 *          wishes to save the weights, the weights are saved to the specified file.
 */
void reportTrainingResults()
{
   cout << "\n" << DIVIDER;
   cout << "Training Results:\n";
   cout << DIVIDER;

   cout << "Final Average Error:  " << avgError << endl;
   cout << "Number of Iterations: " << curIterations << endl << endl;
   time(&endTime);
   printTime(difftime(endTime, startTime));

   cout << "Ended training after reaching the: \n";
   if (curIterations >= maxIterations)
      cout << "Maximum Number of Iterations\n";
   if (avgError <= avgErrCut)
      cout << "Average Error Cutoff\n";

   if (isSaving)
   {
      cout << "\nSaving weights to file: " << saveFilename << endl;
      saveWeights();
   }

   return;
} // void reportTrainingResults()

/**
* \brief   Reports the results of the network.
*          If the network is training:
 *         The final average error and number of iterations are reported, along with the truth table, expected output,
 *         and actual output. If the user wishes to save the weights, the weights are saved to the specified file.
 *         If the network is running:
 *         Reports the truth table, expected output, and actual output for each test case.
 */
void reportResults()
{
   if (isTraining)
      reportTrainingResults();

   reportTruthTable();

   return;
} // void reportResults()

/**
 * \brief   Runs the network for all test cases in the truth table and updates the stored activations.
 */
void runAllTestCases()
{
   for (curTestCase = 0; curTestCase < numTrainingSets; ++curTestCase)
   {
      runNetwork();
   }
}  // void runAllTestCases()

/**
 * \brief   Runs the A-B-C neural network, training or running based on the user's specifications. The network first
 *          sets the configuration parameters and presents them to the user, then allocates memory for the arrays used
 *          in either training or running the network. The arrays are then populated with their required values, and the
 *          network is either trained or run. The results are then reported to the user, and if training, the weights
 *          may be saved to a file.
 * \return  0 (ignore output)
 */
int main(int argc, char* argv[])
{
   srand(time(NULL));
   time(&startTime);

   if (argc > 0)
      configFileName = argv[0];
   else
      configFileName = DEFAULT_CONFIG;

   setConfigurationParameters();
   echoConfigurationParameters();

   allocateMemory();
   populateArrays();

   if (isTraining)
   {
      trainNetwork();
      runAllTestCases();
   }
   else
   {
      runAllTestCases();
   }

   reportResults();

   return 0;
} // int main()
