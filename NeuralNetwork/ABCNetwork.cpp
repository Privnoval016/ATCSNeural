/**
 * Author: Pranav Sukesh
 * Date of Creation: 1/26/2024
 * This project is an A-B-1 feed-forward neural network that can be both trained and ran.
 * The user can specify specific configuration parameters for the network in the setConfigurationParameters() function,
 * including the number of input and hidden activation layers, the average error cutoff, the maximum number of
 * iterations, the lambda value, and the type of and range of weight population.
 * The network runs by calculating a weighted sum of the previous activation layer's activations and applying the
 * activation function to the sum, and the error is calculated as the squared difference between the expected and actual
 * values divided by 2. The truth table is reported to the user.
 * The network trains by utilizing gradient descent to minimize error. It does so by first running the network, then
 * calculating the partial derivatives of the error with respect to each individual weight, then applying these changes
 * in weights to the weights. Through continued iterations of this process, the network either reaches the average error
 * cutoff or the maximum number of iterations. The average error, number of iterations, reason for terminating training,
 * and truth table is then reported to the user.
 */

#include <iomanip>
#include <iostream>

using namespace std;


#define DIVIDER "------------------------------------------------------------------------------\n"

#define NUM_IN_ACT         2
#define NUM_HID_ACT        2
#define NUM_OUT_ACT        1

#define TRAINING_NETWORK   false

#define ACT_FUNCT          SIGMOID        // supported functions: SIGMOID, LINEAR

#define LAMBDA             0.3

#define W_POPULATION       RANDOMIZE      // supported types: RANDOMIZE, FILELOAD
#define HI_RANGE           1.5
#define LOW_RANGE          (-1.5)

#define MAX_ITERATIONS     1e5
#define AVG_ERROR          2e-4

#define NUM_TRAINING_SETS  4

typedef double* DARRAY1D;
typedef double** DARRAY2D;

enum FUNCT_TYPE { SIGMOID, LINEAR };             // enum for types of activation functions
enum POP_TYPE { RANDOMIZE, FILELOAD };           // enum for types of weight population

/* general network parameters */

int numInAct, numHidAct, numOutAct;

bool isTraining;

FUNCT_TYPE actFunctName;
double (*actFunct)(double);                     // uses function pointers instead of conditionals to not branch paths
double (*actFunctDeriv)(double);

POP_TYPE weightPopulation;

double lambda, hiRange, lowRange;
int maxIterations, curIterations;
double error, avgError, avgErrCut;
int numTrainingSets, curTestCase;


/* Running Arrays */
DARRAY1D a;                                     // input activations
DARRAY1D h;                                     // hidden activations
DARRAY1D F;                                     // output activation

DARRAY2D kjWeights;                             // weights between the hidden and input activations
DARRAY2D jiWeights;                             // weights between the hidden and output activations

DARRAY2D truthTable;
DARRAY2D expectedValues;

/* Training Specific Arrays */
DARRAY1D jThetas;
DARRAY1D iThetas;

DARRAY1D iOmegas;
DARRAY1D jOmegas;

DARRAY1D iPsis;
DARRAY1D jPsis;

DARRAY2D jiPartialDeriv;
DARRAY2D kjPartialDeriv;

DARRAY2D jiDeltaWeights;
DARRAY2D kjDeltaWeights;


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

/**
 * \brief   Sets the configuration parameters for the network. These parameters include:
 *          The number of input, hidden, and output activations (along with accumulators for each)
 *          Whether the network is training or running,
 *          The type of weight population,
 *          The range of the random weights,
 *          The lambda value applied in training,
 *          The error cutoff for training (and its accumulator),
 *          The maximum allowed iterations when training (and its accumulator),
 *          And the number of specified training sets,
 */
void setConfigurationParameters()
{
   numInAct = NUM_IN_ACT;
   numHidAct = NUM_HID_ACT;
   numOutAct = NUM_OUT_ACT;

   isTraining = TRAINING_NETWORK;

   actFunctName = ACT_FUNCT;

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

   weightPopulation = W_POPULATION;

   hiRange = HI_RANGE;
   lowRange = LOW_RANGE;

   lambda = LAMBDA;
   avgErrCut = AVG_ERROR;
   error = avgError = 0.0;

   maxIterations = MAX_ITERATIONS;
   numTrainingSets = NUM_TRAINING_SETS;
   curIterations = curTestCase = 0;

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
   cout << numInAct << "-" << numHidAct << "-" << numOutAct << " Network Configuration Parameters\n";
   cout << DIVIDER;
   cout << "Running Mode: " << (isTraining ? "Training" : "Running") << " Network\n";
   cout << DIVIDER;
   if (isTraining)
   {
      cout << "Runtime Training Parameters:\n";
      cout << "Random range for weights: [" << lowRange << ", " << hiRange << "]\n";
      cout << "Max iterations:           " << maxIterations << endl;
      cout << "Error threshold:          " << avgErrCut << endl;
      cout << "Lambda Value:             " << lambda << endl;
      cout << DIVIDER;
   }
   return;
} // void echoConfigurationParameters()

/**
 * \brief   Allocates memory for the arrays used in the network, including:
 *          The input/hidden activation arrays and the output activation,
 *          The weights between the hidden and input activations and the hidden and output activations,
 *          The truth table and expected values for the training sets,
 *          And training-specific arrays (if the network is training), including:
 *             The thetas (a weighted sum of the previous layer's activations),
 *             The omega values (the difference between the expected and actual output),
 *             The psi values (omega * theta evaluated in the derivative of the activation function),
 *             The partial derivatives of the error with respect to the weights,
 *             And the delta weights (the change in weights that will be applied for the next iteration),
 */
void allocateMemory()
{
   int i, j, k;

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
      iThetas = new double[numOutAct];

      jPsis = new double[numHidAct];
      iPsis = new double[numOutAct];

      jOmegas = new double[numHidAct];
      iOmegas = new double[numOutAct];

      kjPartialDeriv = new DARRAY1D[numHidAct];
      for (j = 0; j < numHidAct; ++j) kjPartialDeriv[j] = new double[numInAct];
      jiPartialDeriv = new DARRAY1D[numHidAct];
      for (i = 0; i < numOutAct; ++i) jiPartialDeriv[i] = new double[numHidAct];

      kjDeltaWeights = new DARRAY1D[numHidAct];
      for (j = 0; j < numHidAct; ++j) kjDeltaWeights[j] = new double[numInAct];
      jiDeltaWeights = new DARRAY1D[numHidAct];
      for (i = 0; i < numOutAct; ++i) jiDeltaWeights[i] = new double[numHidAct];
   } // if (isTraining)

   return;
} // void allocateMemory()

/**
 * \brief   Populates the arrays utilized in the network, that were created in the allocateMemory() function.
 *          The truth table is populated with combinations of 0 and 1, and the expected value is changed by the user to
 *          run different bitwise operators. Additionally, the weights can be populated with either randomized values
 *          or loaded based on user specifications (currently set to values for a 2-2-1 network).
 */
void populateArrays()
{
   int i, j, k;

   truthTable[0][0]  = 0.0;              // Inputs for the truth table
   truthTable[0][1]  = 0.0;

   truthTable[1][0]  = 0.0;
   truthTable[1][1]  = 1.0;

   truthTable[2][0]  = 1.0;
   truthTable[2][1]  = 0.0;

   truthTable[3][0]  = 1.0;
   truthTable[3][1]  = 1.0;

   expectedValues[0][0] = 0.0;               // User inputs expected values for specific operators here
   expectedValues[1][1] = 1.0;
   expectedValues[2][2] = 1.0;

   switch (weightPopulation)
   {
      case RANDOMIZE:                      // Populates the weights with random values
         for (j = 0; j < numHidAct; ++j)
         {
            for (k = 0; k < numInAct; ++k)
            {
               kjWeights[j][k] = randDouble();
            }
         }
         for (i = 0; i < numOutAct; ++i)
         {
            for (j = 0; j < numHidAct; ++j)
            {
               kjWeights[i][j] = randDouble();
            }
         }
         break;
      case FILELOAD:                       // Loads weights for a 2-2-3 network
         kjWeights[0][0] = -4.1093;
         kjWeights[1][0] = 3.67544;

         kjWeights[0][1] = -3.73903;
         kjWeights[1][1] = -7.76941;

         jiWeights[0][0] = -8.71564;
         jiWeights[1][0] = 8.73887;
         jiWeights[0][1] = -14.5858;
         jiWeights[1][1] = 6.49954;
         jiWeights[0][2] = 4.4009;
         jiWeights[1][2] = -578.322;
         break;
   } // switch (weightPopulation)

   return;
} // void populateArrays()

/**
 * \brief   Runs the network for the current test case and calculates the error. The network calculates the weighted
 *          sum of the input activations and applies the activation function to the sum to determine the values for
 *          each hidden activation. This process is repeated for the output activation, and the error is calculated
 *          as the squared difference between the expected and actual output divided by 2. The theta and omega values
 *          are stored locally.
 */
void runNetwork()
{
   int i, j, k;

   double iTheta, jTheta, omega;

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
      omega = expectedValues[curTestCase][i] - F[i];
      error += omega * omega * 0.5;
   } // for (i = 0; i < numOutAct; ++i)

   return;
} // void runNetwork()

/**
 * \brief   Runs the network for the current test case and calculates the error. This function is called while training
 *          the network, so it stores the theta values and omegas in their respective global arrays.
 */
void runForTrain()
{
   int i, j, k;

   a = truthTable[curTestCase];

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
      iThetas[i] = 0.0;
      for (j = 0; j < numHidAct; ++j)
      {
         iThetas[i] += h[j] * jiWeights[i][j];      // weighted sum of the hidden activations
      }
      F[i] = actFunct(iThetas[i]);
      iOmegas[i] = expectedValues[curTestCase][i] - F[i];
      error += iOmegas[i] * iOmegas[i] * 0.5;
   } // for (i = 0; i < numOutAct; ++i)

   return;
} // void runNetwork()

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

   for (curTestCase = 0; curTestCase < numTrainingSets; ++curTestCase)     // calculates initial average error
   {
      runForTrain();
      avgError += error;
   }
   avgError /= numTrainingSets;

   while (avgError > avgErrCut && curIterations < maxIterations)           // trains the network
   {
      curIterations++;

      for (curTestCase = 0; curTestCase < numTrainingSets; ++curTestCase)
      {
         for (i = 0; i < numOutAct; ++i)                                   // calculates changes in weights
         {
            iPsis[i] = iOmegas[i] * actFunctDeriv(iThetas[i]);
            for (j = 0; j < numHidAct; ++j)
            {
               jiPartialDeriv[i][j] = -h[j] * iPsis[i];
               jiDeltaWeights[i][j] = -lambda * jiPartialDeriv[i][j];
            }
         } // for (i = 0; i < numOutAct; ++i)

         for (j = 0; j < numHidAct; ++j)
         {
            for (i = 0; i < numOutAct; ++i)
            {
               jOmegas[j] += iPsis[i] * jiWeights[i][j];
            }
            jPsis[j] = jOmegas[j] * actFunct(jThetas[j]);

            for (k = 0; k < numInAct; ++k)
            {
               kjPartialDeriv[j][k] = -a[k] * jPsis[j];
               kjDeltaWeights[j][k] = -lambda * kjPartialDeriv[j][k];
            }
         } // for (j = 0; j < numHidAct; ++j)

         for (j = 0; j < numHidAct; ++j)                    // applies the changes in weights
         {
            for (k = 0; k < numInAct; ++k)
            {
               kjWeights[j][k] += kjDeltaWeights[j][k];
            }
         } // for (j = 0; j < numHidAct; ++j)

         for (i = 0; i < numOutAct; ++i)
         {
            for (j = 0; j < numHidAct; ++j)
            {
               jiWeights[i][j] += jiDeltaWeights[i][j];
            }
         } // for (i = 0; i < numOutAct; ++i

         runForTrain();
         avgError += error;
      } // for (curTestCase = 0; curTestCase < numTrainingSets; ++curTestCase)
      avgError /= numTrainingSets;

   } // while (avgError > avgErrCut && curIterations < maxIterations)

   return;
} // void trainNetwork()

/**
 * \brief   Reports the results of the network after running on the truth table with the current weights. The user is
 *          presented with the truth table, expected output, actual output, and error for each test case.
 */
void runOnTruthTable()
{
   int i, k;

   cout << DIVIDER;
   cout << "Truth Table: \n";

   for (k = 0; k < numInAct; ++k)
   {
      cout << "Input " << k << " | ";
   }
   cout << "Expected Output | Actual Output | Error\n";

   for (curTestCase = 0; curTestCase < numTrainingSets; ++curTestCase)
   {
      runNetwork();
      cout << fixed << setprecision(5) << truthTable[curTestCase][0] << " | ";
      cout << truthTable[curTestCase][1] << " | ";
      cout << "(";
      for (i = 0; i < numOutAct; ++i)
      {
         cout << fixed << setprecision(10) << expectedValues[curTestCase][i] << ", ";
      }
      cout << ") | (";
      for (i = 0; i < numOutAct; ++i)
      {
         cout << fixed << setprecision(10) << F[i] << ", ";
      }
      cout << ") | ";
      cout << error << endl;
   } // for (curTestCase = 0; curTestCase < numTrainingSets; ++curTestCase)
   cout << DIVIDER;

   return;
} // void runOnTruthTable()

/**
 * \brief   Reports the results of the network after training
 *          The final average error and number of iterations are reported, along with the truth table, expected output,
 *          actual output, and error for each test case when running the network with the optimized weights.
 */
void reportTrainingResults()
{

   cout << DIVIDER;
   cout << "Training Results:\n";
   cout << DIVIDER;

   cout << "Final Average Error:  " << avgError << endl;
   cout << "Number of Iterations: " << curIterations << endl;

   cout << "Ended training after reaching the ";
   if (curIterations >= maxIterations) cout << "maximum number of iterations.\n";
   else cout << "average error cutoff.\n";

   runOnTruthTable();

   return;
} // void reportTrainingResults()

/**
 * \brief   Runs the A-B-1 neural network, training or running based on the user's specifications. The network first
 *          sets the configuration parameters and presents them to the user, then allocates memory for the arrays used
 *          in either training or running the network. The arrays are then populated with their required values, and the
 *          network is either trained or run. The results are then reported to the user.
 * \return  0 (ignore output)
 */
int main()
{
   srand(time(NULL));

   setConfigurationParameters();
   echoConfigurationParameters();

   allocateMemory();
   populateArrays();

   if (isTraining)
   {
      trainNetwork();
      reportTrainingResults();
   }
   else
      runOnTruthTable();

   return 0;
} // int main()
