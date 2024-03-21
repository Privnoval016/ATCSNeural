/**
 * Author: Pranav Sukesh
 * Date of Creation: 3/8/2024
 * This file defines the namespace parser, which is used to interpret config files and data files that are used to
 * configure the neural network with the appropriate parameters and data.
 */

#ifndef FILEPARSER_H
#define FILEPARSER_H


#include <iostream>
#include <fstream>
#include <map>

using namespace std;

namespace parser
{
   #define COMMENT_CHAR    '#'
   #define TOKEN_CHAR      '$'
   #define EXT_DATA_CHAR   '?'

   typedef double* DARRAY1D;
   typedef double** DARRAY2D;

   inline string configFileName;

   inline ifstream configStream;
   inline ifstream truthTableStream;

   inline map<string, string> parsedData;

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
      while (trimmedString[index] == ' ')
      {
         index++;
      }
      trimmedString = trimmedString.substr(index);

      index = trimmedString.length() - 1;
      while (trimmedString[index] == ' ')
      {
         index--;
      }
      trimmedString = trimmedString.substr(0, index + 1);

      return trimmedString;

   } // inline string trim(string str)

/**
 *
 * @param stream
 * @return
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
 *
 */
   inline void getNextParam()
   {
      string extractedLine, location, paramValue;
      int index = 0;

      extractedLine = getValidLine(configStream);

      if (!extractedLine.empty() && extractedLine[0] == TOKEN_CHAR)  // if the line is a parameter line
      {
         extractedLine = trim(extractedLine.substr(1, extractedLine.length()-1));

         location = extractedLine.substr(0, extractedLine.find(' '));
         paramValue = trim(extractedLine.substr(extractedLine.find(' ') + 1, extractedLine.length()-1));

         parsedData.insert(pair(location, paramValue));
      } // if (!extractedLine.empty() && extractedLine[0] == TOKEN_CHAR)

      return;
   } // inline void getNextParam()

/**
 *
 * @param kjWeights
 * @param jiWeights
 */
   inline void loadWeightsFromFile(DARRAY2D kjWeights, DARRAY2D jiWeights)
   {
      int i, j, k;
      int weightInAct, weightHidAct, weightOutAct;

      weightInAct = weightHidAct = weightOutAct = 0;

      bool unmatchingConfig;

      if (parsedData["LOAD_FILE"].empty())   // checks if the file to load from is specified
      {
         cout << "\nError: no load file path specified in config. Please provide a valid file path.\n";
         exit(1);
      }

      ifstream loadStream(parsedData["LOAD_FILE"], ios::in | ios::binary);

      if (!loadStream)  // checks if the file to load from exists
      {
         cout << "\nError: file could not be found. Please provide a valid file path.\n";
         exit(1);
      }

      loadStream.read(reinterpret_cast<char*>(&weightInAct), sizeof(int));
      loadStream.read(reinterpret_cast<char*>(&weightHidAct), sizeof(int));
      loadStream.read(reinterpret_cast<char*>(&weightOutAct), sizeof(int));

      unmatchingConfig = weightInAct != stoi(parsedData["NUM_IN_ACT"])
                      || weightHidAct != stoi(parsedData["NUM_HID_ACT"])
                      || weightOutAct != stoi(parsedData["NUM_OUT_ACT"]);

      if (unmatchingConfig)   // checks if the configuration of the file matches the configuration of the network
      {
         cout << "\nError: The specified file does not contain the correct configuration for the network.\n";
         exit(1);
      }

      for (k = 0; k < stoi(parsedData["NUM_IN_ACT"]); ++k)  // for each weight between input and hidden activations
      {
         for (j = 0; j < stoi(parsedData["NUM_HID_ACT"]); ++j)
         {
            loadStream.read(reinterpret_cast<char*>(&kjWeights[j][k]), sizeof(kjWeights[j][k]));
         }
      } // for (k = 0; k < numInAct; ++k)

      for (j = 0; j < stoi(parsedData["NUM_HID_ACT"]); ++j)   // for each weight between hidden and output activations
      {
         for (i = 0; i < stoi(parsedData["NUM_OUT_ACT"]); ++i)
         {
            loadStream.read(reinterpret_cast<char*>(&jiWeights[i][j]), sizeof(jiWeights[i][j]));
         }
      } // for (j = 0; j < numHidAct; ++j)

      loadStream.close();
      return;

   } // inline void loadWeightsFromFile(DARRAY2D kjWeights, DARRAY2D jiWeights)

/**
 *
 * @param kjWeights
 * @param jiWeights
 */
   inline void saveWeightsToFile(DARRAY2D kjWeights, DARRAY2D jiWeights)
   {
      int i, j, k;
      int weightInAct, weightHidAct, weightOutAct;

      if (parsedData["SAVE_FILE"].empty())   // checks if the file to save to is specified
      {
         cout << "\nError: no save file path specified in config. Please provide a valid file path.\n";
         exit(1);
      }

      ofstream saveStream(parsedData["SAVE_FILE"], ios::out | ios::binary);

      if (!saveStream)  // checks if the file to save to exists
      {
         cout << "\nError: file to save to could not be found. Please provide a valid file path.\n";
         exit(1);
      }

      weightInAct = stoi(parsedData["NUM_IN_ACT"]);      // saves the configuration of the network for verification
      weightHidAct = stoi(parsedData["NUM_HID_ACT"]);
      weightOutAct = stoi(parsedData["NUM_OUT_ACT"]);

      saveStream.write(reinterpret_cast<char*>(&weightInAct), sizeof(int));
      saveStream.write(reinterpret_cast<char*>(&weightHidAct), sizeof(int));
      saveStream.write(reinterpret_cast<char*>(&weightOutAct), sizeof(int));

      for (k = 0; k < stoi(parsedData["NUM_IN_ACT"]); ++k)     // for each weight between input and hidden activations
      {
         for (j = 0; j < stoi(parsedData["NUM_HID_ACT"]); ++j)
         {
            saveStream.write(reinterpret_cast<char*>(&kjWeights[j][k]), sizeof(kjWeights[j][k]));
         }
      } // for (k = 0; k < numInAct; ++k)

      for (j = 0; j < stoi(parsedData["NUM_HID_ACT"]); ++j)    // for each weight between hidden and output activations
      {
         for (i = 0; i < stoi(parsedData["NUM_OUT_ACT"]); ++i)
         {
            saveStream.write(reinterpret_cast<char*>(&jiWeights[i][j]), sizeof(jiWeights[i][j]));
         }
      } // for (j = 0; j < numHidAct; ++j)

      saveStream.close();

      return;
   } // inline void saveWeightsToFile(DARRAY2D kjWeights, DARRAY2D jiWeights)

/**
 *
 * @param outputArray
 * @param numElements
 * @param filePath
 */
   inline void loadArrayFromExternalDataFile(DARRAY1D outputArray, int numElements, string filePath)
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
         outputArray[curElement] = stod(extractedLine.substr(0, extractedLine.find(' ')));
         extractedLine = extractedLine.substr(extractedLine.find(' ') + 1, extractedLine.length()-1);
      }

      dataStream.close();
   }  // inline void loadFromExternalDataFile(DARRAY1D outputArray, int numElements, string filePath)

/**
 *
 * @param values
 * @param numElements
 * @param outputArray
 */
   inline void parseTruthTableInfo(string values, int numElements, DARRAY1D outputArray)
   {
      int curElement;
      if (values[0] != EXT_DATA_CHAR)      // if the values are not to be loaded from an external file
      {
         for (curElement = 0; curElement < numElements; ++curElement)
         {
            outputArray[curElement] = stod(values.substr(0, values.find(' ')));
            values = values.substr(values.find(' ') + 1, values.length()-1);
         }
      }
      else
      {
         loadArrayFromExternalDataFile(outputArray, numElements, values.substr(1, values.length()-1));
      }

      return;
   }  // inline void parseTruthTableInfo(string values, int numValues, DARRAY1D outputArray)

/**
 * Methods that are called by the neural network to load information from files
 */

/**
 *
 * @param truthTable
 * @param expectedValues
 */
   inline void loadTruthTableFromFile(DARRAY2D truthTable, DARRAY2D expectedValues)
   {

      int numInputs, numOutputs;
      int i, k, curTestCase;
      string extractedLine, inputValues, outputValues;

      if (parsedData["TRUTH_TABLE_FILE"].empty())     // checks if the truth table file name is specified
      {
         cout << "\nError: no truth table file name specified in config. Please provide a valid file path.\n";
         exit(1);
      }

      truthTableStream.open(parsedData["TRUTH_TABLE_FILE"]);

      if (!truthTableStream)                            // checks if the truth table file exists
      {
         cout << "\nError: truth table file could not be found. Please provide a valid file path.\n";
         exit(1);
      }

      for (curTestCase = 0; curTestCase < stoi(parsedData["NUM_TEST_CASES"]); ++curTestCase) // for each test case
      {
         extractedLine = getValidLine(truthTableStream);

         inputValues = extractedLine.substr(0, extractedLine.find('|') - 1);
         outputValues = extractedLine.substr(extractedLine.find('|') + 2, extractedLine.length()-1);

         parseTruthTableInfo(inputValues, stoi(parsedData["NUM_IN_ACT"]), truthTable[curTestCase]);

         parseTruthTableInfo(outputValues, stoi(parsedData["NUM_OUT_ACT"]), expectedValues[curTestCase]);

      } // for (curTestCase = 0; curTestCase < numTestCases; ++curTestCase)

      truthTableStream.close();

      return;
   }  // inline void loadTruthTableFromFile(DARRAY2D truthTable, DARRAY2D expectedValues)

/**
 *
 * @param name
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

/**
 *
 * @param paramName
 * @return
 */
   inline string getParameter(string paramName)
   {
      return parsedData[paramName];
   }  // inline string getParameter(string paramName)


} // namespace parser

#endif //FILEPARSER_H
