#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include "cppflow/ops.h"
#include "cppflow/model.h"
#include "cppflow/cppflow.h"
#include "sklearn/preprocessing.h"



cppflow::model load_model(std::string model_address){
    cppflow::model model(model_address);
    return model;
}

std::vector<float> prediction_nn(std::vector<float> data, cppflow::model model,
                std::string model_input, std::string model_output){
    auto input = cppflow::tensor(data, {int(data.size()/7),7});
    auto output = model({{model_input, input}},{model_output});
    auto t1 = output[0].get_tensor();
    auto raw_data = static_cast<float*>(TF_TensorData(t1.get()));
    // convert the output tensor into a vector form
    std::vector<float> pred_return;
    for (int i=0; i< int(data.size()/7); i++){
        pred_return.push_back(raw_data[i]);
    }
    return pred_return;
}



int main() {
    
    std::vector<float> data1 = {-33.28, 21.85, 2.03, 1.66, 0.03, 13.3003, 37.4165}; // 0.0106
    std::vector<float> data2 = {-16.02, 2.77, 2.97, 0.96, 0.66, 9.26538, 18.6777}; // 0.6738
    std::vector<float> data3 = {31.57, 33.78, -0.68, 1.46, 0.08, 3.06437, 2.85009}; // 0.893

    // assumming we have 3 data need to be predicted
    std::vector<std::vector<float>> matrix;
    matrix.push_back(data1);
    matrix.push_back(data2);
    matrix.push_back(data3);

    // convert the data matrix into the vector form
    std::vector<float> input_data;
    float data;
    for (int i = 0; i < matrix.size(); i++) {
        for (int j =0; j< 7; j++) {
            // scale data into (-1.0 - 1.0) range
            if (j == 2) {
                data = matrix[i][j]/ 3.1416;
            } else if (j == 3) {
                data = matrix[i][j]/ 2.0;
            } else if (j == 4) {
                data = matrix[i][j]/ 1.48353;
            } else {
                data = matrix[i][j]/ 40.0;
            }
            input_data.push_back(data);
        }
    }
    std::cout << input_data[0] << std::endl;
    // model address
    std::string model1 = "../regress_rrt_env1_20k_0563_350450100";
    std::string model2 = "../regress_rrt_env5_20k_0822_350450100";
    std::string model3 = "../regress_rrt_env6_20k_0886_350450100";
    std::string model4 = "../regress_rrt_env7_20k_0704_350450100";

    std::string model5 = "../regress_rrt_env1_50k_0644_350450100";
    std::string model6 = "../regress_rrt_env5_50k_0826_350450100";
    std::string model7 = "../regress_rrt_env6_50k_0877_350450100";
    std::string model8 = "../regress_rrt_env7_50k_0719_350450100";
    
    std::string model9 = "../regress_rrt_env1_100k_0646_350450100";
    std::string model10 = "../regress_rrt_env5_100k_0830_350450100";
    std::string model11 = "../regress_rrt_env6_100k_0880_350450100";
    std::string model12 = "../regress_rrt_env7_100k_0712_350450100";
    

    // load the model
    cppflow::model model_rrt1_20 = load_model(model1);
    cppflow::model model_rrt5_20 = load_model(model2);
    cppflow::model model_rrt6_20 = load_model(model3);
    cppflow::model model_rrt7_20 = load_model(model4);

    cppflow::model model_rrt1_50 = load_model(model5);
    cppflow::model model_rrt5_50 = load_model(model6);
    cppflow::model model_rrt6_50 = load_model(model7);
    cppflow::model model_rrt7_50 = load_model(model8);

    cppflow::model model_rrt1_100 = load_model(model9);
    cppflow::model model_rrt5_100 = load_model(model10);
    cppflow::model model_rrt6_100 = load_model(model11);
    cppflow::model model_rrt7_100 = load_model(model12);


    std::string input_rrt1_20 = "serving_default_dense_8_input:0";
    std::string input_rrt5_20 = "serving_default_dense_8_input:0";
    std::string input_rrt6_20 = "serving_default_dense_8_input:0";
    std::string input_rrt7_20 = "serving_default_dense_input:0";

    std::string input_rrt1_50 = "serving_default_dense_input:0";
    std::string input_rrt5_50 = "serving_default_dense_input:0";
    std::string input_rrt6_50 = "serving_default_dense_input:0";
    std::string input_rrt7_50 = "serving_default_dense_input:0";

    std::string input_rrt1_100 = "serving_default_dense_8_input:0";
    std::string input_rrt5_100 = "serving_default_dense_input:0";
    std::string input_rrt6_100 = "serving_default_dense_input:0";
    std::string input_rrt7_100 = "serving_default_dense_8_input:0";    

    std::string model_output = "StatefulPartitionedCall:0";

    // make the prediction
    std::vector<float> pred = prediction_nn(input_data, model_rrt1_20, input_rrt1_20, model_output);
    // print the prediction out
    for (int i=0; i< pred.size(); i++) {
        std::cout << pred[i]  << std::endl;    
    }
    
    pred = prediction_nn(input_data, model_rrt5_20, input_rrt5_20, model_output);
    // print the prediction out
    for (int i=0; i< pred.size(); i++) {
        std::cout << pred[i]  << std::endl;    
    }

    pred = prediction_nn(input_data, model_rrt6_20, input_rrt6_20, model_output);
    // print the prediction out
    for (int i=0; i< pred.size(); i++) {
        std::cout << pred[i]  << std::endl;    
    }

    pred = prediction_nn(input_data, model_rrt7_20, input_rrt7_20, model_output);
    // print the prediction out
    for (int i=0; i< pred.size(); i++) {
        std::cout << pred[i]  << std::endl;    
    }

    pred = prediction_nn(input_data, model_rrt1_50, input_rrt1_50, model_output);
    // print the prediction out
    for (int i=0; i< pred.size(); i++) {
        std::cout << pred[i]  << std::endl;    
    }
    
    pred = prediction_nn(input_data, model_rrt5_50, input_rrt5_50, model_output);
    // print the prediction out
    for (int i=0; i< pred.size(); i++) {
        std::cout << pred[i]  << std::endl;    
    }

    pred = prediction_nn(input_data, model_rrt6_50, input_rrt6_50, model_output);
    // print the prediction out
    for (int i=0; i< pred.size(); i++) {
        std::cout << pred[i]  << std::endl;    
    }

    pred = prediction_nn(input_data, model_rrt7_50, input_rrt7_50, model_output);
    // print the prediction out
    for (int i=0; i< pred.size(); i++) {
        std::cout << pred[i]  << std::endl;    
    }

    // make the prediction
    pred = prediction_nn(input_data, model_rrt1_100, input_rrt1_100, model_output);
    // print the prediction out
    for (int i=0; i< pred.size(); i++) {
        std::cout << pred[i]  << std::endl;    
    }
    
    pred = prediction_nn(input_data, model_rrt5_100, input_rrt5_100, model_output);
    // print the prediction out
    for (int i=0; i< pred.size(); i++) {
        std::cout << pred[i]  << std::endl;    
    }

    pred = prediction_nn(input_data, model_rrt6_100, input_rrt6_100, model_output);
    // print the prediction out
    for (int i=0; i< pred.size(); i++) {
        std::cout << pred[i]  << std::endl;    
    }

    pred = prediction_nn(input_data, model_rrt7_100, input_rrt7_100, model_output);
    // print the prediction out
    for (int i=0; i< pred.size(); i++) {
        std::cout << pred[i]  << std::endl;    
    }


    return 0;
}
