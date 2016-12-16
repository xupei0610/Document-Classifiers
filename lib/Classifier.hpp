//
//  Classifier.hpp
//  Document Classifier using cosine similarity.
//  Define the macro RIDGE to obtain ridge regression binary classifier.
//  Otherwise to obtain Rocchio/Centroid-based classifier.
//
//  Created by Pei Xu on 11/23/16.
//  Copyright © 2016 Pei Xu. All rights reserved.
//
//  @version 0.9
//  @updated Dec.7 2016
//

#ifndef Classifier_hpp
#define Classifier_hpp

#include <iostream>
#include <iomanip>
#include <string>
#include <deque>
#include <queue>
#include <map>
#include <unordered_map>
#include <numeric>
#include <chrono>
#include <iterator>
#include <limits>
#include <exception>

#include "Eigen/Dense"
#include "Eigen/Sparse"

namespace PX {

#ifdef RIDGE
    class ClassifierExceptionInvalidWeightCoefficient: public std::exception
    {
        virtual const char* what() const throw()
        {
            return "Invaild weight coefficient provided to classifier.";
        }
    } invaild_weight_coeff_except;
#endif

    template<class ObjectID, class ClassID, class ScalarT>
    class Classifier
    {
    private:
        struct _Test_Point
        {
            const ObjectID id;
            const std::deque<int> attribute;
            const std::deque<ScalarT> value;

            std::unordered_map<ClassID, ScalarT> dist;
            // For ridge regression classifier
            //   it is the prediction value computed by the regression coefficient obtained
            // For centroid-based classifier,
            //   it is the dissimilarity from the test point to the centroid of the class to which this point is assigned
            // This variable is for calculating prediction score when classificaiton evaluation

#ifndef RIDGE
            Eigen::SparseVector<ScalarT> vec;
#endif
            _Test_Point(const ObjectID & id, const std::deque<int> & attribute, const std::deque<ScalarT> & value) : id(id),attribute(attribute), value(value) {}
        };
        struct _Training_Point
        {
            const ClassID label;
            const std::deque<int> attribute;
            const std::deque<ScalarT> value;
            _Training_Point(const ClassID & label, const std::deque<int> & attribute, const std::deque<ScalarT> & value) : label(label), attribute(attribute), value(value) {}
        };
        int _dim = -1;
        std::unordered_map<ClassID, std::deque<_Training_Point *> > _training_pts;
        std::unordered_map<ObjectID, _Test_Point *> _test_pts;
#ifdef RIDGE
        std::unordered_map<ClassID, std::vector<ScalarT> > _weight_coeff;
#else
        struct _Centroid
        {
            const ClassID label;
            Eigen::Matrix<ScalarT, Eigen::Dynamic, 1> vec;
            Eigen::Matrix<ScalarT, Eigen::Dynamic, 1> neg_vec;
            _Centroid(const ClassID & label, const Eigen::Matrix<ScalarT, Eigen::Dynamic, 1> & vec) : label(label), vec(vec) {}
        };
        std::unordered_map<ClassID, _Centroid *> _centroids;
#endif
    public:
        ~Classifier()
        {
            for (const auto & c : this->_training_pts)
                for (auto p : c.second)
                    delete p;
            for (const auto & p : this->_test_pts)
                delete p.second;
#ifndef RIDGE
            for (const auto & c : this->_centroids)
                delete c.second;
#endif
        }
        int addTrainingDataObject(const ClassID & class_id, const std::deque<int> & attribute, const std::deque<ScalarT> & value)
        {
            if (attribute.empty())
                return 1;       // Empty point
            if (attribute.size() != value.size())
                return 2;       // Different number of attributes and values
            // Update the max dimension if needed
            int max_dim = *std::max_element(attribute.begin(), attribute.end());
            if (max_dim > this->_dim)
                this->_dim = max_dim;
            // Record the new point
            auto ptr = this->_training_pts.find(class_id);
            if (ptr == this->_training_pts.end())
                this->_training_pts[class_id] = std::deque<_Training_Point *>({new _Training_Point(class_id, attribute, value)});
            else
                ptr->second.push_front(new _Training_Point(class_id, attribute, value));
            return 0;
        }
        int addTestDataObject(const ObjectID & object_id, const std::deque<int> & attribute, const std::deque<ScalarT> & value)
        {
            if (attribute.empty())
                return 1;       // Empty point
            if (attribute.size() != value.size())
                return 2;       // Different number of attributes and values
            if (this->_test_pts.find(object_id) != this->_test_pts.end())
                return 3;       // Repeated point
            // Update the max dimension if needed
#ifndef RIDGE
            int max_dim = *std::max_element(attribute.begin(), attribute.end());
            if (max_dim > this->_dim)
                this->_dim = max_dim;
#endif
            // Record the new point
            this->_test_pts[object_id] = new _Test_Point(object_id, attribute, value);
            return 0;
        }
#ifdef RIDGE
        // Return a map recording each token's weight
        void train(const ScalarT & regularization_parameter, const ScalarT & tol_threshold, const int & max_iteration)
        {
#else
        void train(const bool & preparation_for_evaluation = false)
        {
#endif
            auto time = std::chrono::high_resolution_clock::now();

            int i;
            int max_dim = this->_dim+1;

#ifdef RIDGE
            this->_weight_coeff.clear();
            std::unordered_map<int, std::unordered_map<int, ScalarT> > x_mat_map_t;

            std::unordered_map<ClassID, int> class_id_map;
            std::unordered_map<int, ClassID> class_inverse_id_map;
            typename std::unordered_map<ClassID, int>::iterator class_id_map_it;

            for (i = max_dim; --i>-1;)
                x_mat_map_t[i] = std::unordered_map<int, ScalarT>();

            int obj = 0;
            int total_class = 0;
            ScalarT norm;
            for (const auto & c : this->_training_pts)
            {
                for (const auto & p : c.second)
                {
                    class_id_map_it = class_id_map.find(p->label);
                    if (class_id_map_it == class_id_map.end())
                    {
                        class_inverse_id_map[total_class] = p->label;
                        class_id_map[p->label] = total_class++;
                    }

                    norm = std::sqrt(std::inner_product(p->value.begin(), p->value.end(), p->value.begin(), 0.0));
                    for (i = p->attribute.size(); --i > -1;)
                        x_mat_map_t[p->attribute[i]][obj] = p->value[i]/norm;
                    obj++;
                }
            }


            Eigen::SparseMatrix<ScalarT, Eigen::ColMajor> temp_x_mat(obj, max_dim);
            temp_x_mat.setZero();
            for (i = 0; i < max_dim; i++)
            {
                // this is squared norm
                norm = std::accumulate(x_mat_map_t[i].begin(), x_mat_map_t[i].end(), 0.0,
                                               [](ScalarT x, std::pair<int, ScalarT> y)
                                               {
                                                   return x + y.second * y.second;
                                               }
                );
                norm += regularization_parameter;
                for (auto & v : x_mat_map_t[i])
                {
                    temp_x_mat.insert(v.first, i) = v.second;
                    v.second /= norm;
                }
            }

            Eigen::SparseMatrix<ScalarT, Eigen::RowMajor> x_mat(std::move(temp_x_mat));
            std::vector<ScalarT> y_mat(obj, -1);
            Eigen::Matrix<ScalarT, Eigen::Dynamic, 1> w(max_dim);

            ScalarT last_w_i, new_w_i, error, stop_cri = tol_threshold * tol_threshold, last_norm;
            int iter, classifier_num = 0, obj_indx = 0;

            std::chrono::time_point<std::chrono::high_resolution_clock> iter_time;

            for (const auto & c : this->_training_pts)
            {
                std::clog << "  Generating Binary Classifier #" << ++classifier_num << std::endl;
                std::fill(y_mat.begin(), y_mat.end(), -1);
                for (i = c.second.size(); --i > -1;)
                    y_mat[obj_indx++] = 1;

                w.setZero();
                this->_weight_coeff[c.first] = std::vector<ScalarT>(max_dim, 0);
                last_norm = 0;
                for (iter = 0; iter < max_iteration;)
                {
                    iter_time = std::chrono::high_resolution_clock::now();
                    error = 0;
                    for (i = 0; i <= this->_dim; i++)
                    {
                        last_w_i = w[i];
                        w[i] = 0;
                        new_w_i = 0;
                        for (const auto & o : x_mat_map_t[i])
                            new_w_i += o.second*(y_mat[o.first]-x_mat.row(o.first)*w);

                        if (new_w_i < 0)
                        {
                            error += last_w_i*last_w_i;
                            this->_weight_coeff[c.first][i] = 0;
                        }
                        else
                        {
                            error += (last_w_i-new_w_i) * (last_w_i-new_w_i);
                            this->_weight_coeff[c.first][i] = new_w_i;
                            w[i] = new_w_i;
                        }
                    }

                    std::clog << "  Iteration: "  << ++iter
                    << ". Squared Error: " << std::scientific << error
                    << ". Time Taken: " << std::fixed << std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1> > > (std::chrono::high_resolution_clock::now() - iter_time).count() << "s" << std::endl;

                    if (error < stop_cri * last_norm)
                        break;

                    last_norm = w.squaredNorm();
                }
            }

#else
            ScalarT norm;
            int total_training_objs = 0;
            for (const auto & c : this->_training_pts)
            {
                this->_centroids[c.first] = new _Centroid(c.first, Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>(max_dim));
                this->_centroids[c.first]->vec.setZero();
                for (auto p : c.second)
                {
                    // Normalize into unit-length, l2-norm
                    norm = std::sqrt(std::inner_product(p->value.begin(), p->value.end(), p->value.begin(), 0.0));
                    for (i = p->attribute.size(); --i > -1;)
                        this->_centroids[c.first]->vec.coeffRef(p->attribute[i]) += p->value[i]/norm;
                    total_training_objs++;
                }
                if (preparation_for_evaluation == false)
                {
                    this->_centroids[c.first]->vec /= c.second.size();
                    this->_centroids[c.first]->vec /= this->_centroids[c.first]->vec.norm();
                }
            }

            if (preparation_for_evaluation == true)
            {
                for (const auto & c : this->_centroids)
                {
                    c.second->neg_vec = Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>(max_dim);
                    c.second->neg_vec.setZero();
                    for (const auto & cc : this->_centroids)
                    {
                        if (cc.second->label == c.second->label)
                            continue;
                        c.second->neg_vec.noalias() += cc.second->vec;
                    }
                    c.second->neg_vec /= total_training_objs - this->_training_pts[c.second->label].size();
                    c.second->neg_vec.normalize();
                }
                for (const auto & c : this->_centroids)
                {
                    c.second->vec /= this->_training_pts[c.second->label].size();
                    c.second->vec.normalize();
                }
            }

#endif
            std::clog << "Training completed. Total Time Taken: " << std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1> > > (std::chrono::high_resolution_clock::now() - time).count() << "s" << std::endl;
        }
        std::unordered_map<ObjectID, ClassID> classify()
        {
            auto time = std::chrono::high_resolution_clock::now();
            std::unordered_map<ObjectID, ClassID> result;
#ifdef RIDGE
            result = this->classify(this->_weight_coeff);
#else
            int i, max_dim = this->_dim + 1;
            Eigen::SparseVector<ScalarT> vec(max_dim);
            ScalarT min_dis_sim, dis_sim;
            _Centroid * target_centroid = nullptr;
            for (const auto & p : this->_test_pts)
            {
                // Vectorize data
                vec.setZero();
                for (i = p.second->attribute.size(); --i > -1;)
                    vec.insert(p.second->attribute[i]) = p.second->value[i];
                vec /= vec.norm();
                p.second->vec = vec;

                // Find the most similar, least dissimilar centroid
                min_dis_sim = 3;
                for (const auto & c : this->_centroids)
                {
                    dis_sim = 1 - vec.dot(c.second->vec);
                    if (dis_sim < min_dis_sim)
                    {
                        min_dis_sim = dis_sim;
                        target_centroid = c.second;
                    }
                    p.second->dist[c.first] = dis_sim;
                }
                result[p.second->id] = target_centroid->label;
            }
#endif
            std::clog << "Classification completed. Total Time Taken: " << std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1> > > (std::chrono::high_resolution_clock::now() - time).count() << "s" << std::endl;

            return result;
        }
#ifdef RIDGE
        std::unordered_map<ObjectID, ClassID> classify(const std::unordered_map<ClassID, std::vector<ScalarT> > & weight_coeff)
        {
            std::unordered_map<ObjectID, ClassID> result;
            ScalarT prob, max_prob, squared_norm;
            ClassID target_class_label;
            int i, max_dim = this->_dim + 1;
            for (const auto & w : weight_coeff)
            {
                if (w.second.size() != max_dim)
                {
                    throw invaild_weight_coeff_except;
                    return result;
                }
            }
            for (const auto & p : this->_test_pts)
            {
                max_prob = std::numeric_limits<ScalarT>::lowest();
                for (const auto & w : weight_coeff)
                {
                    prob = 0;
                    squared_norm = 0;
                    for (i = p.second->attribute.size(); --i > -1;)
                        if (p.second->attribute[i] < max_dim)
                        {
                            squared_norm += p.second->value[i] * p.second->value[i];
                            prob += p.second->value[i] * w.second[p.second->attribute[i]];
                        }
                    prob /= std::sqrt(squared_norm);
                    p.second->dist[w.first] = prob;
                    if (prob > max_prob)
                    {
                        max_prob = prob;
                        target_class_label = w.first;
                    }
                }
                result[p.first] = target_class_label;
            }
            return result;
        }
#endif

        void evaluate(const std::unordered_map<ObjectID, ClassID> & object_class_map, const std::unordered_map<ObjectID, ClassID> & solution, std::ostream *output_stream)
        {
            typename std::unordered_map<ObjectID, ClassID>::const_iterator ptr;
            auto ptr_end = object_class_map.end();
            ScalarT pred_score = 0;
#ifdef RIDGE
            typename std::unordered_map<ClassID, std::deque<_Training_Point *> >::iterator class_ptr;
            auto class_ptr_end = this->_training_pts.end();
#else
            typename std::unordered_map<ClassID, _Centroid *>::iterator class_ptr;
            auto class_ptr_end = this->_centroids.end();
#endif

            std::map<ClassID, std::map<ClassID, int> > conf_mat;
            std::map<ClassID, std::vector<int> > evals;  // tp, fp, fn
            int total_test_objects = 0;
            int max_digit = 16; // size of "Prediction Score"
            for (const auto & c : this->_training_pts)
            {
                if (c.first.size() > max_digit)
                    max_digit = c.first.size();
                evals[c.first] = std::vector<int>({0, 0, 0});
                conf_mat[c.first] = std::map<ClassID, int>();
                for (auto cc : this->_training_pts)
                    conf_mat[c.first][cc.first] = 0;
            }
            int total_class = evals.size();

            for (const auto & p : solution)
            {
                ptr = object_class_map.find(p.first);
                if (ptr == ptr_end)
                {
                    std::clog << "Ignore an object without class information provided. Document ID: " << p.first << std::endl;
                    continue;
                }
                if (p.second != ptr->second)
                {
#ifdef RIDGE
                    class_ptr = this->_training_pts.find(ptr->second);
#else
                    class_ptr = this->_centroids.find(ptr->second);
#endif
                    if (class_ptr == class_ptr_end)
                    {
                        std::clog << "Ignore an object whose class is not provided by any training data. Document ID: " << p.first << std::endl;
                        continue;
                    }
                    evals[ptr->second][2]++;    // fn
                    evals[p.second][1]++;       // fp
#ifndef RIDGE
                    pred_score += 1 - this->_test_pts[p.first]->dist[ptr->second] - this->_test_pts[p.first]->vec.dot(this->_centroids[ptr->second]->neg_vec);
#endif
                }
                else
                {
                    evals[p.second][0]++;       // tp
#ifndef RIDGE
                    pred_score += 1 - this->_test_pts[p.first]->dist[p.second] - this->_test_pts[p.first]->vec.dot(this->_centroids[ptr->second]->neg_vec);
#endif
                }
#ifdef RIDGE
                pred_score += this->_test_pts[p.first]->dist[ptr->second];
#endif
                conf_mat[p.second][ptr->second] += 1;
                total_test_objects++;
            }

            if (total_test_objects == 0)
            {
                std::clog << "No vaild classification solution found." << std::endl;
                return;
            }

            *output_stream << "\nConfusion Matrix:\n" << std::setw(max_digit/2+4) << "Predicted" << "\t\t\tActual";

            for (const auto & c : conf_mat)
            {
                *output_stream << "\n" << std::setw(max_digit) << c.first << "\t";
                for (const auto & cc : c.second)
                    *output_stream << cc.second << "\t";
            }


            *output_stream << "\n\nEvaluation:\n" << std::setw(max_digit) << " " << "\tAccuracy\tPrecision\tRecall   \tF1\t\t\tMax F1 Score as a pure binary classifier\n";
            ScalarT ppv, sen, acc, f1;
            int total_tp = 0, total_fp = 0, total_fn = 0;
            ScalarT macro_ppv = 0, macro_sen = 0, macro_acc = 0, macro_f1 = 0, macro_max_f1 = 0;
            auto max_f1 = evalMaxPossibleF1Score(object_class_map);
            for (const auto & c : evals)
            {
                acc = (ScalarT)(total_test_objects - c.second[1] - c.second[2])/total_test_objects;
                ppv = (ScalarT)c.second[0]/(c.second[0]+c.second[1]);
                sen = (ScalarT)c.second[0]/(c.second[0]+c.second[2]);
                f1  = ppv*sen/(ppv+sen) * 2;
                macro_ppv += ppv;
                macro_sen += sen;
                macro_acc += acc;
                macro_f1  += f1;
                macro_max_f1 += max_f1[c.first];
                *output_stream << std::setw(max_digit) << c.first << "\t" << std::fixed << acc << "\t" << std::fixed << ppv << "\t" << std::fixed << sen << "\t" << std::fixed << f1 << "\t" << std::fixed << max_f1[c.first] << "\n";
                total_tp += c.second[0];
                total_fp += c.second[1];
                total_fn += c.second[2];
            }

            ppv = (ScalarT)total_tp/(total_tp + total_fp);
            sen = (ScalarT)total_tp/(total_tp+total_fn);
            *output_stream << "\n" << std::setw(max_digit) << "Micro-Averaged" << "\t" << std::fixed << (ScalarT)(total_class * total_test_objects -  total_fp - total_fn)/(total_class*total_test_objects) << "\t" << std::fixed << ppv << "\t" << std::fixed << sen << "\t" << std::fixed << ppv*sen/(ppv+sen)*2 << "\n"
            << std::setw(max_digit) << "Macro-Averaged" << "\t" << std::fixed << macro_acc/total_class << "\t" << std::fixed << macro_ppv/total_class << "\t" << std::fixed << macro_sen/total_class << "\t" << std::fixed << macro_f1/total_class << "\t" << std::fixed << macro_max_f1/total_class << "\n" << "\n"
            << std::setw(max_digit) << "Prediction Score" << "\t" << pred_score << "\t" << pred_score/total_test_objects << "(avg. per test object)"
            << std::endl;
        }

        // First element is the micro-averaged F1
        std::pair<ScalarT, std::unordered_map<ClassID, ScalarT> >
        evalScore(
                    const std::unordered_map<ObjectID, ClassID> & object_class_map,
                    const std::unordered_map<ObjectID, ClassID> & solution
        )
        {

            std::pair<ScalarT, std::unordered_map<ClassID, ScalarT> > result;

            if (solution.size() == 0)
            {
                result.first = std::numeric_limits<ScalarT>::quiet_NaN();
                return result;
            }

            std::map<ClassID, std::vector<int> > evals;  // tp, fp, fn
            for (const auto & c : this->_training_pts)
                evals[c.first] = std::vector<int>({0, 0, 0});

            typename std::unordered_map<ObjectID, ClassID>::const_iterator ptr;
            auto ptr_end = object_class_map.end();
            typename std::unordered_map<ClassID, std::deque<_Training_Point *> >::iterator class_ptr;
            auto class_ptr_end = this->_training_pts.end();
            auto set_ptr_end = this->_test_pts.end();

            for (const auto & p : solution)
            {
                if (this->_test_pts.find(p.first) == set_ptr_end)
                    continue;
                ptr = object_class_map.find(p.first);
                if (ptr == ptr_end)
                    continue;
                if (p.second != ptr->second)
                {
                    class_ptr = this->_training_pts.find(ptr->second);
                    if (class_ptr == class_ptr_end)
                        continue;
                    evals[ptr->second][2]++;    // fn
                    evals[p.second][1]++;       // fp
                }
                else
                    evals[p.second][0]++;       // tp
            }

            int total_tp = 0, total_fp = 0, total_fn = 0;
            for (const auto & c : evals)
            {
                result.second[c.first] = 2.0 * c.second[0] / (2 * c.second[0] + c.second[1] + c.second[2]);
                total_tp += c.second[0];
                total_fp += c.second[1];
                total_fn += c.second[2];
            }
            result.first = 2.0 * total_tp / (2 * total_tp + total_fn + total_fp);
            return result;
        }


        // This is a strange function
        // It computes the max possible F1 score each classifier as a pure binary calssifier can achieve.
        // That is, for a binary classifier, try to set different threshold, all test objects whose similarity to this classifer is greater than the threhold is considered to be assigned to the class that this classifier represents, then compute the F1 score. Finally, the max possible F1 score this classifier can achieve is the maximum value among all the F1 scores compuated.
        // In fact, each classifier of the final classification solution may reach a higher value.
        // That is because, in order to ensure every test object must be assigned into a class, a test object always is assigned to the class closest to it. That is to say, even if the similarity between a test object and a classifier is quite small, the test object still may be assigned into the class that classifier represents, due to that the similarity between this test object and other classifiers are smaller.
        // And, of course, it may be higher than the value obtained through the final classification solution.
        // So, for a classifier, this so-called 'max possible F1 score' is the max F1 socre the classifier as a pure binary classifier can achieve.
        std::unordered_map<ClassID, ScalarT> evalMaxPossibleF1Score(const std::unordered_map<ObjectID, ClassID> & object_class_map)
        {
            std::unordered_map<ClassID, ScalarT> result;
            int total_pts = this->_test_pts.size();
            std::vector<std::pair<ClassID, ScalarT> > sim_vec;
            sim_vec.reserve(total_pts);

            int tp, tn;
            ScalarT max_f1, f1;
#ifdef RIDGE
            for (auto & c : this->_training_pts)
#else
            for (auto & c : this->_centroids)
#endif
            {
                sim_vec.clear();
                for (auto & p : this->_test_pts)
                    sim_vec.push_back(
                        std::make_pair(
                                       object_class_map.find(p.first)->second,
#ifdef RIDGE
                                       p.second->dist[c.first]
#else
                                       1 - p.second->dist[c.first] - p.second->vec.dot(c.second->neg_vec)
#endif
                                       )
                    );

                max_f1 = 0;
                for (auto & s : sim_vec)
                {
                    tp = 0;
                    tn = 0;
                    for (auto & ss : sim_vec)
                    {
                        if (ss.second > s.second)
                        {
                            if (ss.first == c.first)
                                tp++;
                        }
                        else if(ss.first != c.first)
                            tn++;
                    }
                    f1 = 2.0*tp/(tp+total_pts-tn);
                    if (f1 > max_f1)
                        max_f1 = f1;
                }

                result[c.first] = max_f1;
            }
            return result;
        }

        struct min_heap_cmp
        {
            bool operator ()(const std::pair<int, ScalarT> & x, const std::pair<int, ScalarT> & y){return x.second>y.second;}
        };

        std::map<ClassID, std::deque<std::pair<int, ScalarT> > > getHighWeightedAttributes(const int & first_k)
        {
            std::map<ClassID, std::deque<std::pair<int, ScalarT> > > result;

            int max_dim = this->_dim + 1;
            int key_indx = max_dim - first_k;

            std::priority_queue<
                std::pair<int, ScalarT>,
                std::vector<std::pair<int, ScalarT> >,
                min_heap_cmp
            > heap;

            ScalarT * data;
            ScalarT min_max = 0.0;

#ifdef RIDGE
            for (auto & c : this->_weight_coeff)
            {
                data = &(c.second[0]) + this->_dim;
#else
            Eigen::Matrix<ScalarT, Eigen::Dynamic, 1> dif_vec(max_dim);
            for (const auto & c : this->_centroids)
            {
                dif_vec = c.second->vec - c.second->neg_vec;
                data = dif_vec.data() + this->_dim;
#endif
                std::priority_queue<
                    std::pair<int, ScalarT>,
                    std::vector<std::pair<int, ScalarT> >,
                    min_heap_cmp
                >().swap(heap);
                for (int i= max_dim;--i>-1;data--)
                {
                    if (i >= key_indx)
                    {
                        heap.push(std::make_pair(i, *data));
                        if (i == key_indx)
                            min_max = heap.top().second;
                    }
                    else if (*data > min_max)
                    {
                        heap.pop();
                        heap.push(std::make_pair(i, *data));
                        min_max = heap.top().second;
                    }
                }

                result[c.first] = std::deque<std::pair<int, ScalarT> >();
                while (!heap.empty())
                {
                    if (heap.top().second == 0)
                        continue;
                    result[c.first].push_front(std::move(heap.top()));
                    heap.pop();
                }
            }

            return result;
        }
#ifdef RIDGE
            auto getWeightCoefficient() -> decltype(this->_weight_coeff)
            {
                return this->_weight_coeff;
            }
#else
            std::unordered_map<ClassID, std::vector<ScalarT> > getWeightCoefficient()
            {
                std::unordered_map<ClassID, std::vector<ScalarT> > weight_coeff;
                int max_dim = this->_dim + 1;
                ScalarT * data;
                for (const auto & c : this->_centroids)
                {
                    weight_coeff[c.first] = std::vector<ScalarT>();
                    weight_coeff[c.first].reserve(max_dim);
                    data = c.second->vec.data();
                    for (int i = 0; i < max_dim ; i++, data++)
                        weight_coeff[c.first].push_back(*data);
                }
                return weight_coeff;
            }
#endif

    };

}


#endif /* Classifier_hpp */
