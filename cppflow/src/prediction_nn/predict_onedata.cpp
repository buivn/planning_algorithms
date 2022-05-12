#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include "cppflow/ops.h"
#include "cppflow/model.h"
#include "cppflow/cppflow.h"


float prediction_nn(std::vector<float> data, std::string model_address){
    auto input = cppflow::tensor(data, {1,5});
    cppflow::model model(model_address);
    std::cout << input << std::endl;
    auto output = model({{"serving_default_inputs:0", input}},{"StatefulPartitionedCall:0"});
    auto t1 = output[0].get_tensor();
    auto raw_data = static_cast<float*>(TF_TensorData(t1.get()));
    float result_value = raw_data[0];
    return result_value;
    // return 0;
    // std::cout << result_value << std::endl;
}



int main() {

    std::vector<float> data = {-3.55576, 4.21125, 2.29887, 1.18671, -1.02827};
    std::string model1 = "../model";
    std::cout << prediction_nn(data, model1) << std::endl;


    return 0;
}
