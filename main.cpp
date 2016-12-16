//
//  main.cpp
//  Main program for performing document classification.
//  Load data and run classification.
//
//  Created by Pei Xu  on 11/23/16.
//  Copyright © 2016 Pei Xu. All rights reserved.
//
//  @version: 0.9
//  @updated: Dec.7 2016

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <deque>
#include <map>
#include <set>
#include <unordered_map>
#include <tuple>
#include <limits>

#include "lib/Classifier.hpp"

typedef std::string DocumentCodeName;
typedef std::string ClassCodeName;
typedef std::string TokenCodeName;

#define EMPTY_DOC_ID ""
#define DELIM " "

#ifdef RIDGE
  #ifndef MAX_ITERATION
    #define MAX_ITERATION 25
  #endif
  #ifndef CONVERGENCE_TOL
    #define CONVERGENCE_TOL 0.01 // this is a realtively value
  #endif
  #ifndef DEFAULT_REGULAR_PARAMETER
    #define DEFAULT_REGULAR_PARAMETER 0.01
  #endif
  #ifndef CANDIDATE_REGULAR_PARAMETERS
    #define CANDIDATE_REGULAR_PARAMETERS 0.01,0.05,0.1,0.5,1.0,10.0
  #endif
#endif

#ifndef SHOW_HIGH_WEIGHTED_TOKEN
#define SHOW_HIGH_WEIGHTED_TOKEN 10
#endif

enum FrequencyRepresentForm
{
    TF,
    BINARY,
    TFIDF
};

static const char * FrequencyRepresentName[] = {"Term Frequency", "Binary Frequency", "TF-IDF"};

const char * get_frequency_represent_name(const FrequencyRepresentForm & represent_form)
{
    return FrequencyRepresentName[represent_form];
}

struct Document
{
    const DocumentCodeName id;
    int flag;         // -1 training, 0 both, 1 test
    std::map<TokenCodeName, int> tokens;
    Document(const DocumentCodeName & id, const int & flag) : id(id), flag(flag) {}
    Document(const int & flag) : id(EMPTY_DOC_ID), flag(flag) {}
};

inline std::string trim(const std::string & str)
{
    size_t start = str.find_first_not_of(" ");
    if (start == std::string::npos)
        return "";
    else
        return str.substr(0, str.find_last_not_of(" ")+1).substr(start);
}

inline const std::string get_file_contents(std::ifstream * file_handle)
{
    std::string contents;
    contents.resize(file_handle->seekg(0, std::ios::end).tellg());
    file_handle->seekg(0, std::ios::beg).read(&contents[0], static_cast<std::streamsize>(contents.size()));
    return contents;
}

std::unordered_map<DocumentCodeName, ClassCodeName> parse_class_file(std::ifstream * file_handle)
{
    std::unordered_map<std::string, std::string> doc_class_map;
    std::string entry, sub_string;
    size_t pos;
    
    std::istringstream content_stream(get_file_contents(file_handle));
    while (std::getline(content_stream, entry))
    {
        entry = trim(std::move(entry));
        if (entry.empty())
            continue;
        pos = entry.find_first_of(DELIM);
        if (pos == std::string::npos)
        {
            std::cerr << "Ignore Incompleted document classification information. Class Info.: " << entry << std::endl;
            continue;
        }
        doc_class_map[entry.substr(0, pos)] = entry.substr(pos+1);
    }
    return doc_class_map;
}

std::set<DocumentCodeName> parse_train_file(std::ifstream * file_handle, const std::unordered_map<DocumentCodeName, ClassCodeName> & class_info)
{
    std::set<std::string> result;
    std::string entry;
    
    std::istringstream content_stream(get_file_contents(file_handle));
    while (std::getline(content_stream, entry))
    {
        entry = trim(std::move(entry));
        if (entry.empty())
            continue;
        if (class_info.find(entry) == class_info.end())
            std::cerr << "Ignore Undefined training data document. " << entry << std::endl;
        else
            result.insert(entry);
    }
    return result;
}

std::set<DocumentCodeName> parse_test_file(std::ifstream * file_handle)
{
    std::set<std::string> result;
    std::string entry;
    
    std::istringstream content_stream(get_file_contents(file_handle));
    while (std::getline(content_stream, entry))
    {
        entry = trim(std::move(entry));
        if (!entry.empty())
            result.insert(std::move(entry));
    }
    return result;
}

std::unordered_map<TokenCodeName, std::string> parse_token_file(std::ifstream * file_handle)
{
    std::unordered_map<TokenCodeName, std::string> result;
    std::string entry;
    
    int line_num = 0;
    std::istringstream content_stream(get_file_contents(file_handle));
    while (std::getline(content_stream, entry))
    {
        line_num++;
        entry = trim(std::move(entry));
        if (!entry.empty())
            result[std::to_string(line_num)] = std::move(entry);
    }
    return result;
}

std::tuple<
    std::unordered_map<DocumentCodeName, Document *>,    // Training data set
    std::unordered_map<DocumentCodeName, Document *>,    // Test data set
    std::unordered_map<TokenCodeName, int>          // Term Frequency for each token in taining data used for TFIDF
>
parse_data_file(std::ifstream * file_handle,
                const std::set<std::string> & training_doc_name_set,
                const std::set<std::string> & test_doc_name_set,
                const std::unordered_map<DocumentCodeName, ClassCodeName> & doc_class_map)
{
    std::unordered_map<std::string, Document *> train_data_set;
    std::unordered_map<std::string, Document *> test_data_set;
    std::unordered_map<TokenCodeName, int>      token_freq;

    size_t pos_1, pos_2;
    int val;
    TokenCodeName token_name;
    std::string entry, doc_name;
    std::string last_doc_name = "";
    Document * last_doc = nullptr;
    std::istringstream content_stream(get_file_contents(file_handle));
    while (std::getline(content_stream, entry))
    {
        entry = trim(std::move(entry));
        if (entry.empty())
            continue;
        
        pos_1 = entry.find_first_of(DELIM);
        if (pos_1 == std::string::npos)
        {
            std::cerr << "Ignore Incompleted document data information. Data Info.: " << entry << std::endl;
            continue;
        }
        
        pos_2 = entry.find_last_of(DELIM);
        if (pos_2 <= pos_1)
        {
            std::cerr << "Ignore Incompleted document data information. Data Info.: " << entry << std::endl;
            continue;
        }
        
        val = std::stoi(entry.substr(pos_2+1));
        if (val < 1)
        {
            std::cerr << "Ignore Invaild document token information. Data Info.: " << entry << std::endl;
            continue;
        }
        
        doc_name = entry.substr(0, pos_1);
        token_name = entry.substr(pos_1+1, pos_2-pos_1-1);
        
        if (doc_name != last_doc_name)
        {
            last_doc = nullptr;
            
            if (test_doc_name_set.find(doc_name) != test_doc_name_set.end())
            {
                if (test_data_set.find(doc_name) == test_data_set.end())
                {
                    last_doc = new Document(doc_name, 1);
                    test_data_set[doc_name] = last_doc;
                }
                else
                    last_doc = test_data_set[doc_name];
            }
            
            if (training_doc_name_set.find(doc_name) != training_doc_name_set.end())
            {
                if (train_data_set.find(doc_name) == train_data_set.end())
                {
                    if (doc_class_map.find(doc_name) == doc_class_map.end())
                    {
                        std::cerr << "Ignore Undefined training data document. No classification info. found for " << doc_name << std::endl;
                        last_doc = nullptr;
                    }
                    else
                    {
                        if (last_doc == nullptr)
                            last_doc = new Document(-1);
                        else
                            last_doc->flag = 0;
                        train_data_set[doc_name] = last_doc;
                    }
                }
                else if (last_doc == nullptr)
                    last_doc = train_data_set[doc_name];
            }
            
            if (last_doc == nullptr)
                continue;
        }
        
        if (last_doc->flag < 1)
        {
            if (token_freq.find(token_name) == token_freq.end())
                token_freq[token_name] = 1;
            else if (last_doc->tokens.find(token_name) == last_doc->tokens.end())
                token_freq[token_name] += 1;
        }
        last_doc->tokens[token_name] = val;
    }
    return std::make_tuple(std::move(train_data_set), std::move(test_data_set), std::move(token_freq));
}

std::deque<TokenCodeName> input_training_and_test_doc(
    PX::Classifier<DocumentCodeName, ClassCodeName, double> * classifier,
    const std::unordered_map<DocumentCodeName, Document *> & training_data,
    const std::unordered_map<DocumentCodeName, Document *> & test_data,
    const std::unordered_map<TokenCodeName, int> & token_count,
    const std::unordered_map<DocumentCodeName, ClassCodeName> & class_info,
    const FrequencyRepresentForm & represent_form
)
{
    size_t train_set_size = training_data.size();
    std::unordered_map<TokenCodeName, int> token_name_id_map;
    std::deque<TokenCodeName> token_id_name_map;

    std::deque<int> attr;
    std::deque<double> val;
    std::unordered_map<TokenCodeName, int>::iterator ptr;
    int map_indx = 0;
    if (represent_form == FrequencyRepresentForm::TF)
    {
        for (const auto & tr : training_data)
        {
            attr.clear();
            val.clear();
            for (const auto & tk : tr.second->tokens)
            {
                val.push_front(tk.second);
                ptr = token_name_id_map.find(tk.first);
                if (ptr == token_name_id_map.end())
                {
                    token_name_id_map[tk.first] = map_indx;
                    token_id_name_map.push_back(tk.first);
                    attr.push_front(map_indx++);
                }
                else
                    attr.push_front(ptr->second);
            }
            classifier->addTrainingDataObject(class_info.find(tr.first)->second, std::move(attr), std::move(val));
            if (tr.second->flag < 0)
                delete tr.second;
        }
        for (const auto & te : test_data)
        {
            attr.clear();
            val.clear();
            for (const auto & tk : te.second->tokens)
            {
                val.push_front(tk.second);
                ptr = token_name_id_map.find(tk.first);
                if (ptr == token_name_id_map.end())
                {
                    token_name_id_map[tk.first] = map_indx;
                    token_id_name_map.push_back(tk.first);
                    attr.push_front(map_indx++);
                }
                else
                    attr.push_front(ptr->second);
            }
            classifier->addTestDataObject(std::move(te.second->id), std::move(attr), std::move(val));
            delete te.second;
        }
    }
    else if (represent_form == FrequencyRepresentForm::BINARY)
    {
        for (const auto & tr : training_data)
        {
            attr.clear();
            val.clear();
            for (const auto & tk : tr.second->tokens)
            {
                val.push_front(1);
                ptr = token_name_id_map.find(tk.first);
                if (ptr == token_name_id_map.end())
                {
                    token_name_id_map[tk.first] = map_indx;
                    token_id_name_map.push_back(tk.first);
                    attr.push_front(map_indx++);
                }
                else
                    attr.push_front(ptr->second);
            }
            classifier->addTrainingDataObject(class_info.find(tr.first)->second, std::move(attr), std::move(val));
            if (tr.second->flag < 0)
                delete tr.second;
        }
        for (const auto & te : test_data)
        {
            attr.clear();
            val.clear();
            for (const auto & tk : te.second->tokens)
            {
                val.push_front(1);
                ptr = token_name_id_map.find(tk.first);
                if (ptr == token_name_id_map.end())
                {
                    token_name_id_map[tk.first] = map_indx;
                    token_id_name_map.push_back(tk.first);
                    attr.push_front(map_indx++);
                }
                else
                    attr.push_front(ptr->second);
            }
            classifier->addTestDataObject(std::move(te.second->id), std::move(attr), std::move(val));
            delete te.second;
        }
    }
    else
    {
        for (const auto & tr : training_data)
        {
            attr.clear();
            val.clear();
            for (const auto & tk : tr.second->tokens)
            {
                val.push_front(tk.second*log2(train_set_size/(token_count.find(tk.first)->second)));
                ptr = token_name_id_map.find(tk.first);
                if (ptr == token_name_id_map.end())
                {
                    token_name_id_map[tk.first] = map_indx;
                    token_id_name_map.push_back(tk.first);
                    attr.push_front(map_indx++);
                }
                else
                    attr.push_front(ptr->second);
            }
            classifier->addTrainingDataObject(class_info.find(tr.first)->second, std::move(attr), std::move(val));
            if (tr.second->flag < 0)
                delete tr.second;
        }
        std::unordered_map<TokenCodeName, int>::const_iterator const_ptr;
        for (const auto & te : test_data)
        {
            attr.clear();
            val.clear();
            for (const auto & tk : te.second->tokens)
            {
                const_ptr = token_count.find(tk.first);
                if (const_ptr == token_count.end())
                {
                    val.push_front(0);             // IDF = 0 for undefined token
                    // val.push_front(tk.second);  // IDF = 1 for undefined token
                    ptr = token_name_id_map.find(tk.first);
                    if (ptr == token_name_id_map.end())
                    {
                        token_name_id_map[tk.first] = map_indx;
                        token_id_name_map.push_back(tk.first);
                        attr.push_front(map_indx++);
                    }
                    else
                        attr.push_front(ptr->second);
                }
                else
                {
                    attr.push_front(token_name_id_map[tk.first]);
                    val.push_front(tk.second*log2(train_set_size/(const_ptr->second)));
                }
            }
            classifier->addTestDataObject(std::move(te.second->id), std::move(attr), std::move(val));
            delete te.second;
        }
    }
    return token_id_name_map;
}

void show_high_weight_tokens(
                             const std::map<ClassCodeName, std::deque<std::pair<int, double> > > weights,
                             const std::deque<TokenCodeName> & token_id_name_map,
                             const std::unordered_map<TokenCodeName, std::string> & token_map,
                             const int & show_how_many,
                             std::ostream *output
)
{
    int max_digit = 0;

    for (const auto & c : weights)
        if (c.first.size() > max_digit)
            max_digit = c.first.size();

    if (token_map.empty())
    {
        for (const auto & c : weights)
        {
            *output << std::setw(max_digit) << c.first << "\t" << token_id_name_map[c.second.begin()->first];
            for (int i=1; i < show_how_many; i++)
                *output << ", " << token_id_name_map[c.second[i].first];
            *output << "\n";
        }
    }
    else
    {
        std::unordered_map<TokenCodeName, std::string>::const_iterator ptr;
        auto ptr_end = token_map.end();
        std::string token_name;
        for (const auto & c : weights)
        {
            ptr = token_map.find(token_id_name_map[c.second.begin()->first]);
            if (ptr == ptr_end)
                *output << std::setw(max_digit) << c.first << "\t" << token_id_name_map[c.second.begin()->first] << "(Undefined Token)";
            else
                *output << std::setw(max_digit) << c.first << "\t" << ptr->second;

            for (int i=1; i < show_how_many; i++)
            {
                ptr = token_map.find(token_id_name_map[c.second[i].first]);
                if (ptr == ptr_end)
                    std::cout << ", " << token_id_name_map[c.second[i].first] << "(Undefined token)";
                else
                    std::cout << ", " << ptr->second;
            }
            std::cout << "\n";
        }
    }

    std::cout << std::endl;
}

int main(int argc, char * argv[])
{
    // Each line record the term frequncy of a token who belongs to a document in the following form
    std::string data_file;       // document_id token_id term_frequency_of_token
    
    // It records the file name from which a document is extracted
    std::string rlable_file;
    
    std::string train_file;      // a Document ID of training data for each line
    std::string test_file;       // a Document ID of test data for each line
    
    // Each line record the name of the class to which a document belongs in the following form
    std::string class_file;      // document_id class_name
    // It contains all the documents in both of training and test set.
    // used for both of classifiction evaluation
    
    std::string token_file;   // Each line is the word or token each token_id represents
    
    // The form to represent a document data vector
    // tf for term frequency, tfidf for TFIDF,
    // binary for the case where 1 means the document has that token while 0 means does not have
    FrequencyRepresentForm represent_form;
    
    std::string output_file;     // The file records the classification result

#ifdef RIDGE
    std::string sample_file_train;
    std::string sample_file_test;
        // a small set of training and test objects, used to determine a better regularization parameter before
        // performing classification on the whole group of, real training and test objects.
#endif
    
    int no_evaluation = 0;  // 0 for performing evaluation (default)

    

    if (argc > 8)
    {
        std::string arg_str;
        std::transform(std::make_move_iterator(argv[7]), std::make_move_iterator(argv[7]+strlen(argv[7])), std::back_inserter(arg_str), (int (*)(int))std::toupper);
        if (arg_str == "TF")
            represent_form = FrequencyRepresentForm::TF;
        else if (arg_str == "BINARY")
            represent_form = FrequencyRepresentForm::BINARY;
        else if (arg_str == "TFIDF" || arg_str == "TF-IDF")
            represent_form = FrequencyRepresentForm::TFIDF;
        else
        {
            std::cerr << "Program Stopped.";
            std::cerr << "Error: Unsupported Represetation Option. Now only support TF, BINARY and TFIDF." << std::endl;
            throw;
        }
        data_file   = argv[1];
        rlable_file = argv[2];
        train_file  = argv[3];
        test_file   = argv[4];
        class_file  = argv[5];
#if SHOW_HIGH_WEIGHTED_TOKEN > 0
        token_file  = argv[6];
#endif
        output_file = argv[8];
        if (argc > 9)
        {
#ifdef RIDGE
            sample_file_train = argv[9];
            if (argc > 10)
                sample_file_test = argv[10];
            no_evaluation  = 0;
#else
            no_evaluation = std::atoi(argv[9]);
#endif
        }
    }
    else
    {
        std::cerr << "Program Stopped.";
        std::cerr << "Error: Wrong Number of Arguments provided." << std::endl;
        throw;
    }
    
    
    std::cout << "Parameters:\n  Classifier:      " <<
#ifdef RIDGE
    "Ridge Regression"
#else
    "Centroid-Based"
#endif
    << "\n  Input data file: " << data_file << "\n  Training data:   " << train_file << "\n  Test data:       " << test_file << "\n  Class. data:     " << class_file << "\n  Token data:      " << token_file << "\n  Freq. represent: " << get_frequency_represent_name(represent_form) << "\n  Output file:     " << output_file;
#ifdef RIDGE
    if (!sample_file_train.empty())
    {
        if (sample_file_test.empty())
            std::cout << "\n  Validation File: " << sample_file_train;
        else
            std::cout << "\n  Val.Train. Data: " << sample_file_train << "\n  Val. Test Data:  " << sample_file_test;
    }
    std::cout << "\n  Conv. Tolerance: " << CONVERGENCE_TOL << "\n  Max Iteration:   " << MAX_ITERATION << "\n";
#endif

    std::cout << std::endl;
    
    
    std::clog << "Loading data files..." << std::endl;
    
    std::ifstream file_handle;

    // Load classification information for training data and
    // also for test data if evaluation is needed
    file_handle.open(class_file);
    if (file_handle.fail())
    {
        std::cerr << "Program Stopped.\nError: Unable to Open the Classification Information File. " << class_file << std::endl;
        throw;
    }
    auto class_info = parse_class_file(&file_handle);
    file_handle.close();
    
    // Load Training data
    file_handle.open(train_file);
    if (file_handle.fail())
    {
        std::cerr << "Program Stopped.\nError: Unable to Open the Train Data File. " << train_file << std::endl;
        throw;
    }
    auto training_set = parse_train_file(&file_handle, class_info);
    file_handle.close();
    if (training_set.empty())
    {
        std::cerr << "Program Stopped.";
        std::cerr << "Error: No training data provided." << std::endl;
        throw;
    }
    
    // Load Test data
    std::set<DocumentCodeName> test_set;
    if (train_file == test_file)
    {
        test_set = training_set;
    }
    else
    {
        file_handle.open(test_file);
        if (file_handle.fail())
        {
            std::cerr << "Program Stopped.\nError: Unable to Open the Test Data File. " << test_file << std::endl;
            throw;
        }
        test_set = parse_test_file(&file_handle);
        file_handle.close();
        if (test_set.empty())
        {
            std::cerr << "Program Stopped.";
            std::cerr << "Error: No test data provided." << std::endl;
            throw;
        }
    }
    // Prepare output file
    std::ofstream output(output_file);
    if (output.fail())
    {
        std::cerr << "Program Stopped." << std::endl;
        std::cerr << "Error: Unable to open the output file." << output_file << std::endl;
        throw;
    }


#ifdef RIDGE
    double regular_parameter = DEFAULT_REGULAR_PARAMETER;

    std::set<DocumentCodeName> val_train_set;
    std::set<DocumentCodeName> val_test_set;
    if (!sample_file_train.empty())
    {
        if (sample_file_train == train_file)
        {
            val_train_set = training_set;
        }
        else
        {
            file_handle.open(sample_file_train);
            if (file_handle.fail())
            {
                std::cerr << "Error: Unable to Open the Validation Training File " << sample_file_train << ".\nSkip parameter selection. Using default regularization parameter: " << DEFAULT_REGULAR_PARAMETER << std::endl;
                sample_file_train = nullptr;
            }
            else
            {
                val_train_set = parse_test_file(&file_handle);
                file_handle.close();
                if (val_train_set.empty())
                {
                    std::cerr << "Error: No validation training data provided from " << sample_file_train << ".\nSkip parameter selection. Using default regularization parameter: " << DEFAULT_REGULAR_PARAMETER << std::endl;
                    sample_file_train = nullptr;
                }
            }
        }
    }
    if (!sample_file_train.empty())
    {
        if (sample_file_test.empty() || sample_file_test == sample_file_train)
        {
            val_test_set = val_train_set;
        }
        else if (sample_file_test == test_file)
        {
            val_test_set = test_set;
        }
        else if (sample_file_test == train_file)
        {
            val_test_set = training_set;
        }
        else
        {
            file_handle.open(sample_file_test);
            if (file_handle.fail())
            {
                std::cerr << "Error: Unable to Open the Validation Test File" << sample_file_test << ".\nSkip parameter selection. Using default regularization parameter: " << DEFAULT_REGULAR_PARAMETER << std::endl;
                sample_file_train = nullptr;
            }
            else
            {
                val_test_set = parse_test_file(&file_handle);
                file_handle.close();
                if (val_test_set.empty())
                {
                    std::cerr << "Error: No validation test data provided from " << sample_file_test << ".\nSkip parameter selection. Using default regularization parameter: " << DEFAULT_REGULAR_PARAMETER << std::endl;
                    sample_file_train = nullptr;
                }
            }
        }
    }

#endif

    // Load token frequency file
    // for both training and test data
    file_handle.open(data_file);
    if (file_handle.fail())
    {
        std::cerr << "Program Stopped.\nError: Unable to Open the Input Data File. " << data_file << std::endl;
        throw;
    }

    std::unordered_map<std::string, Document *> training_data;
    std::unordered_map<std::string, Document *> test_data;
    std::unordered_map<TokenCodeName, int> token_count;

#ifdef RIDGE
    std::unordered_map<ClassCodeName, std::vector<double> > weight_coeff;
    if (!sample_file_train.empty())
    {
        std::tie(training_data, test_data, token_count) = parse_data_file(&file_handle, val_train_set, val_test_set, class_info);

        PX::Classifier<DocumentCodeName, ClassCodeName, double> classifier;

        input_training_and_test_doc(&classifier, training_data, test_data,
                                    token_count, class_info, represent_form);

        training_data.clear();
        test_data.clear();
        token_count.clear();

        double best_score = 0, score;
        std::unordered_map<ClassCodeName, double> scores;
        int iter = 0, max_digit = 0;
        std::clog << "Choosing regularization parameter through testing validation data..." << std::endl;
        std::vector<double> lambdas = std::vector<double>({CANDIDATE_REGULAR_PARAMETERS});
        size_t total_lam = lambdas.size();
        bool record_coeff = false;
        if (sample_file_train == train_file)
            record_coeff = true;
        for (const auto & lam : lambdas)
        {
            std::clog << "Regularization parameter: " << lam << " (" << ++iter << "/" << total_lam << ")\n" << "Begining iteration..." << std::endl;
            classifier.train(lam, CONVERGENCE_TOL, MAX_ITERATION);

//            score = classifier.evalF1Score(class_info, classifier.classify()).first;
//            std::cout << "Micro-Averaged F1 score: " << std::fixed << score << std::endl;


            classifier.classify();
            scores = classifier.evalMaxPossibleF1Score(class_info);
            score = 0;
            if (max_digit == 0)
            {
                for (const auto & c : scores)
                    if (c.first.size() > max_digit)
                        max_digit = c.first.size();
            }
            std::cout << "\nRegularizatin Parameter: " << lam  << "\nMax F1 Score as a pure binary classifier:\n";
            for (auto & s : scores)
            {
                score += s.second;
                std::cout << std::setw(max_digit) << s.first << "\t" << std::fixed << s.second << "\n";
            }
            score /= scores.size();
            std::cout << "\n" << std::setw(max_digit) << "Average" << "\t" << std::fixed << score << "\n" << std::endl;

            if (score > best_score || iter == 1)
            {
                regular_parameter = lam;
                best_score = score;
                if (record_coeff == true)
                    weight_coeff = classifier.getWeightCoefficient();
            }
        }
        std::cout << "\nBest regularization parameter found: " << regular_parameter << std::endl;
    }
#endif
    std::tie(training_data, test_data, token_count) = parse_data_file(&file_handle, training_set, test_set, class_info);
    file_handle.close();

    // Load tokens' names for outputing high-wighted tokens
    std::unordered_map<TokenCodeName, std::string> token_map;
    file_handle.open(token_file);
    if (file_handle.fail())
    {
        std::cerr << "Error: Unable to Open the token data File. Skip outputing high-weighted tokens." <<std::endl;
        token_file = nullptr;
    }
    else
    {
        token_map = parse_token_file(&file_handle);
        file_handle.close();
    }

    // Load rlable file
    file_handle.open(rlable_file);
    std::unordered_map<DocumentCodeName, std::string> doc_name;
    if (file_handle.fail())
    {
        std::cerr << "Error: Unable to Open the document name file. Skip parsing document name." <<std::endl;
        rlable_file = nullptr;
    }
    else
    {
        std::string entry;
        std::istringstream content_stream(get_file_contents(&file_handle));
        size_t pos;
        while (std::getline(content_stream, entry))
        {
            entry = trim(std::move(entry));
            if (entry.empty())
                continue;

            pos = entry.find_first_of(DELIM);

            if (pos == std::string::npos)
                continue;
            doc_name[entry.substr(0, pos)] = entry.substr(entry.find_last_of(DELIM)+1);

        }
        file_handle.close();
    }

    PX::Classifier<DocumentCodeName, ClassCodeName, double> classifier;
    
    // Add training and test data into classifier
    auto token_id_name_map = input_training_and_test_doc(&classifier, training_data, test_data,
                                token_count, class_info, represent_form);

    // Classify
    std::unordered_map<DocumentCodeName, ClassCodeName> s;
    std::map<ClassCodeName, std::deque<std::pair<int, double> > > w;

#ifdef RIDGE
    if (weight_coeff.empty())
    {
        std::clog << "Training the classifier..." << std::endl;
        classifier.train(regular_parameter, CONVERGENCE_TOL, MAX_ITERATION);
        weight_coeff = classifier.getWeightCoefficient();
    }
    std::clog << "Classifying the test documents..." << std::endl;
    s = classifier.classify(std::move(weight_coeff));
#else
    std::clog << "Training the classifier..." << std::endl;
    classifier.train(true);
    std::clog << "Classifying the test documents..." << std::endl;
    s = classifier.classify();
#endif
    
    if (no_evaluation == 0)
    {
        // Classification Evaluation
        std::clog << "Performing classification evaluation..." << std::endl;
        classifier.evaluate(class_info, s, &std::cout);
    }
    
    if (SHOW_HIGH_WEIGHTED_TOKEN > 0)
    {
        std::cout << "\nHigh-Weighted Tokens:\n\n";
        show_high_weight_tokens(classifier.getHighWeightedAttributes(SHOW_HIGH_WEIGHTED_TOKEN),
                                token_id_name_map, token_map, SHOW_HIGH_WEIGHTED_TOKEN, &std::cout);
    }
    
    // Output solution
    std::clog << "Collecting classification solution..." << std::endl;
    struct classcomp {
        bool operator() (const std::string & lhs, const std::string & rhs) const
        {return std::stoi(lhs)<std::stoi(rhs);}
    };
    std::map<DocumentCodeName, ClassCodeName, classcomp> solution(std::make_move_iterator(s.begin()), std::make_move_iterator(s.end()));
    
    if (doc_name.empty())
        for (const auto & p : solution)
            output << p.first << "," << p.second << "\n";
    else
    {
        std::unordered_map<DocumentCodeName, std::string>::iterator ptr;
        for (const auto & p : solution)
        {
            ptr = doc_name.find(p.first);
            if (ptr == doc_name.end())
                output << p.first << "(Undefined Doc. Name)," << p.second << "\n";
            else
                output << ptr->second << "," << p.second << "\n";
        }
    }
    output.flush();
    output.close();
    std::clog << "Task Completed." << std::endl;
}
