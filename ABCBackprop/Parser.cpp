/**
 * Author: Pranav Sukesh
 * Date of Creation: 3/8/2024
 * This file defines the namespace parser, which is used to interpret config files and data files that are used to
 * configure the neural network with the appropriate parameters and data. The namespace can read from config files,
 * binary weight files, and truth table files.
 * Config files are formatted as follows: each line that starts with the TOKEN_CHAR constant is a parameter line, and the
 * following key and value are separated by a space.
 * Weight files are binary files that store the network configuration followed by the weights.
 * Truth table files are formatted as follows: each line contains the input values followed by a '|' character and the
 * output values. The values can also be loaded from an external file by specifying the file path with the EXT_DATA_CHAR
 * constant.
 * Any comments (denoted by the COMMENT_CHAR constant) and whitespace are omitted from the files.
 */

#include <iostream>
#include <fstream>

using namespace std;

/**
 * The namespace parser is used to interpret config files and data files that are used to configure the neural network.
 * It is capable of reading from config files, binary weight files, and truth table files. It also contains a number of
 * public variables that are used to store the data that is read from the files. The parser can omit comments and
 * whitespace in said files, and its data can be accessed by the neural network to configure itself.
 */
namespace parser
{
   #define COMMENT_CHAR    '#'
   #define TOKEN_CHAR      '$'
   #define EXT_DATA_CHAR   '?'
   #define VALUE_DELIMITER ' '

   typedef double* DARRAY1D;
   typedef double** DARRAY2D;

   inline string configFileName, truthTableFile, loadFilename, saveFilename;

   inline ifstream configStream;                   // the stream used to read from the config file
   inline ifstream truthTableStream;

   inline bool isTraining, isLoading, isSaving;
   inline int actFunctIndex;                       // sigmoid = 0, linear = 1
   inline int keepAliveInterval, savingInterval, numTrainingSets;
   inline int numInAct, numHidAct, numOutAct;

   inline double lambda, avgErrCut, hiRange, lowRange;
   inline int maxIterations;

/**
 * Creates a new input stream to read from the config file with the given name. If the file does not exist, the
 * program terminates with an error message.
 */
   inline void createConfigStream()
   {
      if (configFileName.empty())
      {
         cout << "\nError: no config file name provided. Please provide a valid file path.\n";
         exit(1);
      }

      configStream.open(configFileName);

      if (!configStream)
      {
         cout << "\nError: config file could not be found. Please provide a valid file path.\n";
         exit(1);
      }

      return;
   } // inline void createConfigStream()


/**
 * Trims the whitespace from the beginning and end of the given string.
 * @param str  the string to be trimmed
 * @return     the trimmed string
 */
   inline string trim(string str)
   {
      int index;
      string trimmedString = str;

      index = 0;
      while (trimmedString[index] == VALUE_DELIMITER)
      {
         index++;
      }
      trimmedString = trimmedString.substr(index);

      index = trimmedString.length() - 1;
      while (trimmedString[index] == VALUE_DELIMITER)
      {
         index--;
      }
      trimmedString = trimmedString.substr(0, index + 1);

      return trimmedString;
   } // inline string trim(string str)

/**
 * Returns the next line in the given stream that is not empty or a comment (defined by the COMMENT_CHAR constant).
 * @param stream  the stream to read from
 * @return        the next valid line in the stream
 */
   inline string getValidLine(ifstream &stream)
   {
      string extractedLine;

      getline(stream, extractedLine);
      extractedLine = trim(extractedLine);

      while (stream.peek() != EOF && (extractedLine.empty() || extractedLine[0] == COMMENT_CHAR))
      {
         getline(stream, extractedLine);
         extractedLine = trim(extractedLine);
      }

      return extractedLine;
   }  // inline string getValidLine(ifstream &stream)

   /**
    * Adds the given key-value pair to the appropriate variable in the namespace. The key is used to determine which
    * variable to add the value to and what type the value must be converted to.
    * @param key     the key that determines which variable to add the value to
    * @param value   the value to be added to the variable
    */
   inline void addParamtoVars(string key, string value)
   {
      if (key == "NUM_IN_ACT")
      {
         numInAct = stoi(value);
      }
      else if (key == "NUM_HID_ACT")
      {
         numHidAct = stoi(value);
      }
      else if (key == "NUM_OUT_ACT")
      {
         numOutAct = stoi(value);
      }
      else if (key == "RUN_MODE")
      {
         isTraining = stoi(value) == 1;
      }
      else if (key == "ACT_FUNCT")
      {
         actFunctIndex = stoi(value);
      }
      else if (key == "LOAD_WEIGHTS")
      {
         isLoading = stoi(value) == 1;
      }
      else if (key == "SAVE_WEIGHTS")
      {
         isSaving = stoi(value) == 1;
      }
      else if (key == "KEEP_ALIVE")
      {
         keepAliveInterval = stoi(value);
      }
      else if (key == "SAVE_INTERVAL")
      {
         savingInterval = stoi(value);
      }
      else if (key == "NUM_TEST_CASES")
      {
         numTrainingSets = stoi(value);
      }
      else if (key == "LAMBDA")
      {
         lambda = stod(value);
      }
      else if (key == "MIN_ERROR")
      {
         avgErrCut = stod(value);
      }
      else if (key == "HI_RANGE")
      {
         hiRange = stod(value);
      }
      else if (key == "LOW_RANGE")
      {
         lowRange = stod(value);
      }
      else if (key == "MAX_ITER")
      {
         maxIterations = stoi(value);
      }
      else if (key == "TRUTH_TABLE_FILE")
      {
         truthTableFile = value;
      }
      else if (key == "LOAD_FILE")
      {
         loadFilename = value;
      }
      else if (key == "SAVE_FILE")
      {
         saveFilename = value;
      }

      return;
   }  // inline void addParamtoVars(string key, string value)

/**
 * Reads the next parameter from the config file and adds it to the appropriate variable in the namespace.
 * Parameters are defined by a line that starts with the TOKEN_CHAR constant, followed by the key and value separated
 * by spaces.
 */
   inline void getNextParam()
   {
      string extractedLine, location, paramValue;

      extractedLine = getValidLine(configStream);

      if (!extractedLine.empty() && extractedLine[0] == TOKEN_CHAR)  // if the line is a parameter line
      {
         extractedLine = trim(extractedLine.substr(1, extractedLine.length()-1));

         location = extractedLine.substr(0, extractedLine.find(VALUE_DELIMITER));
         paramValue = trim(extractedLine.substr(extractedLine.find(VALUE_DELIMITER) + 1, extractedLine.length()-1));

         addParamtoVars(location, paramValue);
      } // if (!extractedLine.empty() && extractedLine[0] == TOKEN_CHAR)

      return;
   } // inline void getNextParam()

/**
 * Reads the weights from the specified file and loads them into the given arrays. The binary file is formatted as
 * follows: the first three integers are the network configuration to verify that the file matches the network; the
 * remaining data is the weights between the input and hidden activations followed by the weights between the hidden
 * and output activations stored as doubles. The weights are loaded into the given arrays in the order they are stored
 * to match the network configuration.
 *
 * @param kjWeights  the array to store the weights between the input and hidden activations
 * @param jiWeights  the array to store the weights between the hidden and output activations
 */
   inline void loadWeightsFromFile(DARRAY2D kjWeights, DARRAY2D jiWeights)
   {
      int i, j, k;
      int weightInAct, weightHidAct, weightOutAct;

      weightInAct = weightHidAct = weightOutAct = 0;

      bool unmatchingConfig;

      if (loadFilename.empty())   // checks if the file to load from is specified
      {
         cout << "\nError: no load file path specified in config. Please provide a valid file path.\n";
         exit(1);
      }

      ifstream loadStream(loadFilename, ios::in | ios::binary);

      if (!loadStream)  // checks if the file to load from exists
      {
         cout << "\nError: file could not be found. Please provide a valid file path.\n";
         exit(1);
      }

      loadStream.read(reinterpret_cast<char*>(&weightInAct), sizeof(int));
      loadStream.read(reinterpret_cast<char*>(&weightHidAct), sizeof(int));
      loadStream.read(reinterpret_cast<char*>(&weightOutAct), sizeof(int));

      unmatchingConfig = weightInAct != numInAct || weightHidAct != numHidAct || weightOutAct != numOutAct;

      if (unmatchingConfig)   // checks if the configuration of the file matches the configuration of the network
      {
         cout << "\nError: The specified file does not contain the correct configuration for the network.\n";
         exit(1);
      }

      for (k = 0; k < numInAct; ++k)  // for each weight between input and hidden activations
      {
         for (j = 0; j < numHidAct; ++j)
         {
            loadStream.read(reinterpret_cast<char*>(&kjWeights[j][k]), sizeof(kjWeights[j][k]));
         }
      } // for (k = 0; k < numInAct; ++k)

      for (j = 0; j < numHidAct; ++j)   // for each weight between hidden and output activations
      {
         for (i = 0; i < numOutAct; ++i)
         {
            loadStream.read(reinterpret_cast<char*>(&jiWeights[i][j]), sizeof(jiWeights[i][j]));
         }
      } // for (j = 0; j < numHidAct; ++j)

      loadStream.close();

      return;
   } // inline void loadWeightsFromFile(DARRAY2D kjWeights, DARRAY2D jiWeights)

/**
 * Saves the weights from the given arrays to the specified file. The binary file is formatted as follows: the first
 * three integers are the network configuration to verify that the file matches the network; the remaining data is the
 * weights between the input and hidden activations followed by the weights between the hidden and output activations
 * stored as doubles. The weights are saved to the file in the order they are stored to match the network configuration.
 *
 * @param kjWeights  the array containing the weights between the input and hidden activations
 * @param jiWeights  the array containing the weights between the hidden and output activations
 */
   inline void saveWeightsToFile(DARRAY2D kjWeights, DARRAY2D jiWeights)
   {
      int i, j, k;

      if (saveFilename.empty())   // checks if the file to save to is specified
      {
         cout << "\nError: no save file path specified in config. Please provide a valid file path.\n";
         exit(1);
      }

      ofstream saveStream(saveFilename, ios::out | ios::binary);

      if (!saveStream)  // checks if the file to save to exists
      {
         cout << "\nError: file to save to could not be found. Please provide a valid file path.\n";
         exit(1);
      }

      saveStream.write(reinterpret_cast<char*>(&numInAct), sizeof(int));   // saves network config for validation
      saveStream.write(reinterpret_cast<char*>(&numHidAct), sizeof(int));
      saveStream.write(reinterpret_cast<char*>(&numOutAct), sizeof(int));

      for (k = 0; k < numInAct; ++k)     // for each weight between input and hidden activations
      {
         for (j = 0; j < numHidAct; ++j)
         {
            saveStream.write(reinterpret_cast<char*>(&kjWeights[j][k]), sizeof(kjWeights[j][k]));
         }
      } // for (k = 0; k < numInAct; ++k)

      for (j = 0; j < numHidAct; ++j)    // for each weight between hidden and output activations
      {
         for (i = 0; i < numOutAct; ++i)
         {
            saveStream.write(reinterpret_cast<char*>(&jiWeights[i][j]), sizeof(jiWeights[i][j]));
         }
      } // for (j = 0; j < numHidAct; ++j)

      saveStream.close();

      return;
   } // inline void saveWeightsToFile(DARRAY2D kjWeights, DARRAY2D jiWeights)

/**
 * Loads the given array with the values from the specified external data file. The values are stored in the file as
 * follows: each value is separated by a space, and the values are stored in the order they are to be loaded into the
 * array. The number of values to be loaded is specified by the numElements parameter. Comments and whitespace are
 * omitted from the file.
 *
 * @param filePath      the path to the external data file
 * @param numElements   the number of values to be loaded
 * @param outputArray   the array to store the values
 */
   inline void loadArrayFromExternalDataFile(string filePath, int numElements, DARRAY1D outputArray)
   {
      int curElement;
      string extractedLine;

      ifstream dataStream(filePath);

      if (!dataStream)                          // checks if the external data file exists
      {
         cout << "\nError: external data file for truth table could not be found. Please provide a valid file path.\n";
         exit(1);
      }

      extractedLine = getValidLine(dataStream);

      for (curElement = 0; curElement < numElements; ++curElement)
      {
         outputArray[curElement] = stod(extractedLine.substr(0, extractedLine.find(VALUE_DELIMITER)));
         extractedLine = extractedLine.substr(extractedLine.find(VALUE_DELIMITER) + 1, extractedLine.length()-1);
      }

      dataStream.close();

      return;
   }  // inline void loadFromExternalDataFile(DARRAY1D outputArray, int numElements, string filePath)

/**
 * Parses the given string to load the values into the given array. The values are stored in the string as follows: each
 * value is separated by a space, and the values are stored in the order they are to be loaded into the array. The number
 * of values to be loaded is specified by the numElements parameter. Comments and whitespace are omitted from the string.
 *
 * @param values        the string containing the values to be loaded
 * @param numElements   the number of values to be loaded
 * @param outputArray   the array to store the values
 */
   inline void parseTruthTableInfo(string values, int numElements, DARRAY1D outputArray)
   {
      int curElement;

      if (values[0] != EXT_DATA_CHAR)      // if the values are not to be loaded from an external file
      {
         for (curElement = 0; curElement < numElements; ++curElement)
         {
            outputArray[curElement] = stod(values.substr(0, values.find(VALUE_DELIMITER)));
            values = values.substr(values.find(VALUE_DELIMITER) + 1, values.length()-1);
         }
      }
      else
      {
         loadArrayFromExternalDataFile( values.substr(1, values.length()-1), numElements, outputArray);
      }

      return;
   }  // inline void parseTruthTableInfo(string values, int numValues, DARRAY1D outputArray)


/**
 * Methods that are called by the neural network to load information from files
 */

/**
 * Loads the truth table from the specified file and stores the input and expected output values in the given arrays.
 * the truth table file is formatted as follows: each line contains the input values followed by a '|' character and the
 * output values. The values are separated by spaces, and the values are stored in the order they are to be loaded into
 * the arrays. If the values are to be loaded from an external file, the file path is specified by the EXT_DATA_CHAR
 * constant followed by the file path. The number of values to be loaded is specified by the number of input and output
 * activations. Comments and whitespace are omitted from the file.
 *
 * @param truthTable       the array to store the input values
 * @param expectedValues   the array to store the expected output values
 */
   inline void loadTruthTableFromFile(DARRAY2D truthTable, DARRAY2D expectedValues)
   {
      int curTestCase;
      string extractedLine, inputValues, outputValues;

      if (truthTableFile.empty())     // checks if the truth table file name is specified
      {
         cout << "\nError: no truth table file name specified in config. Please provide a valid file path.\n";
         exit(1);
      }

      truthTableStream.open(truthTableFile);

      if (!truthTableStream)                            // checks if the truth table file exists
      {
         cout << "\nError: truth table file could not be found. Please provide a valid file path.\n";
         exit(1);
      }

      for (curTestCase = 0; curTestCase < numTrainingSets; ++curTestCase) // for each test case
      {
         extractedLine = getValidLine(truthTableStream);

         inputValues = extractedLine.substr(0, extractedLine.find('|') - 1);
         outputValues = extractedLine.substr(extractedLine.find('|') + 2, extractedLine.length()-1);

         parseTruthTableInfo(inputValues, numInAct, truthTable[curTestCase]);
         parseTruthTableInfo(outputValues, numOutAct, expectedValues[curTestCase]);

      } // for (curTestCase = 0; curTestCase < numTestCases; ++curTestCase)

      truthTableStream.close();

      return;
   }  // inline void loadTruthTableFromFile(DARRAY2D truthTable, DARRAY2D expectedValues)


/**
 * Reads the config file with the given name and stores the parameters in the namespace. The config file is formatted as
 * follows: if a line starts with the TOKEN_CHAR constant, the line is a parameter line and the key and value are separated
 * and stored in the appropriate variable. Comments (denoted by the COMMENT_CHAR constant) and whitespace are omitted
 * from the file.
 *
 * @param name the name of the config file to read from
 */
   inline void readConfigFile(string name)
   {
      configFileName = name;
      createConfigStream();

      while (configStream.peek() != EOF)
      {
         getNextParam();
      }

      configStream.close();

      return;
   } // inline void readConfigFile(string name)


} // namespace parser
