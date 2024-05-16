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
 *
 * Table of Contents:
 * namespace parser
 *   inline void createConfigStream()
 *   inline string trim(string str)
 *   inline string getValidLine(ifstream &stream)
 *   inline void setNumLayers()
 *   inline void addParamToVars(string key, string value)
 *   inline void getNextParam()
 *   inline void loadWeightsFromFile(DARRAY3D weights, int* layerConfig)
 *   inline void saveWeightsToFile(DARRAY3D weights, int* layerConfig)
 *   inline void loadArrayFromExternalDataFile(string filePath, int numElements, DARRAY1D outputArray)
 *   inline void parseTruthTableInfo(string values, int numElements, DARRAY1D outputArray)
 *   inline void loadTruthTableFromFile(DARRAY2D truthTable, DARRAY2D expectedValues)
 *   inline void readConfigFile(string name)
 */

#include <iostream>
#include <fstream>
using namespace std;

namespace parser
{
#define COMMENT_CHAR    '#'
#define TOKEN_CHAR      '$'
#define EXT_DATA_CHAR   '?'
#define VALUE_DELIMITER ' '

#define IN_ACT_LAYER    0

   typedef double* DARRAY1D;
   typedef double** DARRAY2D;
   typedef double*** DARRAY3D;

   inline string configFileName, truthTableFile, loadFilename, saveFilename;

   inline ifstream configStream;                   // the stream used to read from the config file
   inline ifstream truthTableStream;

   inline bool isTraining, isLoading, isSaving, isPrintingInput;
   inline int actFunctIndex;                       // sigmoid = 0, tanh = 1, linear = 2
   inline int keepAliveInterval, savingInterval, numTrainingSets, precision;

   inline int numActLayers, numInAct, numOutAct;
   inline string layerConfigData;

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
         throw invalid_argument("No config file name specified. Please provide a valid file path.");
      }

      configStream.open(configFileName);

      if (!configStream)
      {
         throw invalid_argument("Config file could not be found. Please provide a valid file path.");
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
 * Sets the number of layers in the network based on the layer configuration data, stored as a space separated string.
 */
   inline void setNumLayers()
   {
      numActLayers = 0;
      for (int i = 0; i < layerConfigData.length(); i++)
      {
         if (layerConfigData[i] == VALUE_DELIMITER)
         {
            numActLayers++;
         }
      }
      numActLayers++;

      return;
   }  // inline void setNumLayers()

/**
 * Adds the given key-value pair to the appropriate variable in the namespace. The key is used to determine which
 * variable to add the value to and what type the value must be converted to.
 * @param key     the key that determines which variable to add the value to
 * @param value   the value to be added to the variable
 */
   inline void addParamToVars(string key, string value)
   {
      if (key == "PRECISION")
      {
         precision = stoi(value);
      }
      if (key == "ACT_LAYERS")
      {
         layerConfigData = value;
      }
      else if (key == "RUN_MODE")
      {
         isTraining = stoi(value) == 1;
      }
      else if (key == "OUTPUT_MODE")
      {
         isPrintingInput = stoi(value) == 1;
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
   }  // inline void addParamToVars(string key, string value)

/**
 * Sets the given array with the layer configuration data stored as a space separated string. The number of input and
 * output activations are also stored on the layer configuration data.
 *
 * @precondition         the config file's data must be fully read
 * @param layerConfig    the array to store the layer configuration data
 */
   inline void setLayerConfigData(int* layerConfig)
   {
      int n;

      for (n = 0; n < numActLayers; ++n)
      {
         layerConfig[n] = stoi(layerConfigData.substr(0, layerConfigData.find(VALUE_DELIMITER)));
         layerConfigData = layerConfigData.substr
                 (layerConfigData.find(VALUE_DELIMITER) + 1, layerConfigData.length() - 1);
      }

      n = IN_ACT_LAYER;
      numInAct = layerConfig[n];
      numOutAct = layerConfig[numActLayers - 1];

      return;
   }  // inline void setLayerConfigData(int* layerConfig)

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

         addParamToVars(location, paramValue);
      } // if (!extractedLine.empty() && extractedLine[0] == TOKEN_CHAR)

      return;
   } // inline void getNextParam()

/**
 * Reads the weights from the specified file and loads them into the given weight array. The binary file is formatted as
 * follows: the first integers are the network configuration to verify that the file matches the network; the
 * remaining data is the weights between the activations moving forward through the network stored as doubles. The
 * weights are loaded into the given arrays in the order they are stored to match the network configuration.
 *
 * @param weights      the array to store the weights
 * @param layerConfig  the network configuration to verify the file
 */
   inline void loadWeightsFromFile(DARRAY3D weights, int* layerConfig)
   {
      int j, k, n;
      int actSize;
      bool unmatchingConfig;

      if (loadFilename.empty())   // checks if the file to load from is specified
      {
         throw invalid_argument("No load file path specified in config. Please provide a valid file path.");
      }

      ifstream loadStream(loadFilename, ios::in | ios::binary);

      if (!loadStream)  // checks if the file to load from exists
      {
         throw invalid_argument("File to load from could not be found. Please provide a valid file path.");
      }

      unmatchingConfig = false;

      for (n = 0; n < numActLayers; ++n)
      {
         loadStream.read(reinterpret_cast<char *>(&actSize), sizeof(actSize));
         unmatchingConfig = unmatchingConfig || actSize != layerConfig[n];
      }

      if (unmatchingConfig)
      {
         throw invalid_argument
                 ("Configuration of load file does not match network configuration. Please provide a valid file.");
      }

      for (n = 1; n < numActLayers; ++n)
      {
         for (j = 0; j < layerConfig[n]; ++j)
         {
            for (k = 0; k < layerConfig[n - 1]; ++k)
            {
               loadStream.read(reinterpret_cast<char*>(&weights[n][j][k]), sizeof(weights[n][j][k]));
            }
         } // for (j = 0; j < layerConfig[n]; ++j)
      } // for (n = 1; n < numActLayers; ++n)


      loadStream.close();

      return;
   } // inline void loadWeightsFromFile(DARRAY2D kjWeights, DARRAY2D jiWeights)

/**
 * Saves the weights from the given arrays to the specified file. The binary file is formatted as follows: the first
 * integers are the network configuration to verify that the file matches the network; the remaining data is the
 * weights between the activations moving forward through the network stored as doubles. The weights are saved to the
 * file in the order they are stored to match the network configuration.
 *
 * @param weights      the array containing the weights to be saved
 * @param layerConfig  the network configuration to be saved
 */
   inline void saveWeightsToFile(DARRAY3D weights, int* layerConfig)
   {
      int j, k, n;

      if (saveFilename.empty())   // checks if the file to save to is specified
      {
         throw invalid_argument("No save file path specified in config. Please provide a valid file path.");
      }

      ofstream saveStream(saveFilename, ios::out | ios::binary);

      if (!saveStream)  // checks if the file to save to exists
      {
         throw invalid_argument("File to save to could not be found. Please provide a valid file path.");
      }

      for (n = 0; n < numActLayers; ++n)
      {
         saveStream.write(reinterpret_cast<char*>(&layerConfig[n]), sizeof(layerConfig[n]));
      }

      for (n = 1; n < numActLayers; ++n)
      {
         for (j = 0; j < layerConfig[n]; ++j)
         {
            for (k = 0; k < layerConfig[n - 1]; ++k)
            {
               saveStream.write(reinterpret_cast<char*>(&weights[n][j][k]), sizeof(weights[n][j][k]));
            }
         } // for (j = 0; j < layerConfig[n]; ++j)
      } // for (n = 1; n < numActLayers; ++n)


      saveStream.close();

      return;
   } // inline void saveWeightsToFile(DARRAY2D kjWeights, DARRAY2D jiWeights)

/**
 * Loads the given array with the values from the specified external .txt file. The values are stored in the file as
 * follows: each value is separated by a \n, and the values are stored in the order they are to be loaded into the
 * array. The number of values to be loaded is specified by the numElements parameter. Comments and whitespace are
 * omitted from the file.
 *
 * @param filePath      the path to the external data file
 * @param numElements   the number of values to be loaded
 * @param outputArray   the array to store the values
 */
   inline void loadArrayFromExternalTXTFile(string filePath, int numElements, DARRAY1D outputArray)
   {
      int curElement;
      string extractedLine;

      if (filePath.empty())                      // checks if the file path is specified
      {
         throw invalid_argument("No external data file path specified. Please provide a valid file path.");
      }

      ifstream dataStream(filePath);

      if (!dataStream)                          // checks if the external data file exists
      {
         throw invalid_argument("External data file could not be found. Please provide a valid file path.");
      }

      for (curElement = 0; curElement < numElements; ++curElement)
      {
         extractedLine = getValidLine(dataStream);
         outputArray[curElement] = stod(trim(extractedLine));
      }

      dataStream.close();

      return;
   }  // inline void loadFromExternalTXTFile(DARRAY1D outputArray, int numElements, string filePath)

   /**
 * Loads the given array with the values from the specified external .bin file. The values are stored in the file as
 * follows: The values are stored in the order they are to be loaded into the array. The number of values to be loaded
 * is specified by the numElements parameter.
 *
 * @param filePath      the path to the external data file
 * @param numElements   the number of values to be loaded
 * @param outputArray   the array to store the values
 */
   inline void loadArrayFromExternalBINFile(string filePath, int numElements, DARRAY1D outputArray)
   {
      int curElement;
      string extractedLine;

      if (filePath.empty())                      // checks if the file path is specified
      {
         throw invalid_argument("No external data file path specified. Please provide a valid file path.");
      }

      ifstream dataStream(filePath, ios::out | ios::binary);

      if (!dataStream)                          // checks if the external data file exists
      {
         throw invalid_argument("External data file could not be found. Please provide a valid file path.");
      }

      for (curElement = 0; curElement < numElements; ++curElement)
      {
         dataStream.read(reinterpret_cast<char*>(&outputArray[curElement]), sizeof(outputArray[curElement]));
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
            values = values.substr(values.find(VALUE_DELIMITER) + 1, values.length() - 1);
         }
      }  // if (values[0] != EXT_DATA_CHAR)
      else
      {
         string dataFileName = values.substr(1, values.length() - 1);
         if (dataFileName.substr(dataFileName.length() - 4, 4) == ".txt")
         {
            loadArrayFromExternalTXTFile(dataFileName, numElements, outputArray);
         }
         else
         {
            loadArrayFromExternalBINFile(dataFileName, numElements, outputArray);
         }
      }

      return;
   }  // inline void parseTruthTableInfo(string values, int numValues, DARRAY1D outputArray)

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
         throw invalid_argument("No truth table file name specified. Please provide a valid file path.");
      }

      truthTableStream.open(truthTableFile);

      if (!truthTableStream)                            // checks if the truth table file exists
      {
         throw invalid_argument("Truth table file could not be found. Please provide a valid file path.");
      }

      for (curTestCase = 0; curTestCase < numTrainingSets; ++curTestCase) // for each test case
      {
         extractedLine = getValidLine(truthTableStream);

         inputValues = extractedLine.substr(0, extractedLine.find('|') - 1);
         outputValues = extractedLine.substr(extractedLine.find('|') + 2, extractedLine.length() - 1);


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

      setNumLayers();

      configStream.close();

      return;
   } // inline void readConfigFile(string name)

} // namespace parser
