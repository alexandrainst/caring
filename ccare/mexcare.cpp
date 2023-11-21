#include "libcaring.h"
#include "mex.hpp"
#include "mexAdapter.hpp"
#include <utility>

using namespace matlab::data;
using matlab::mex::ArgumentList;


class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        // Function implementation
        checkArguments(outputs, inputs);
        TypedArray<double> data = std::move(inputs[0]);
        auto n = data.getNumberOfElements();
        int err = care_sum_many(data.release(), n);
        // TODO: Handle errors
        outputs[0] = std::move(data);
    }

    void checkArguments(ArgumentList outputs, ArgumentList inputs) {
        std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
        ArrayFactory factory;
        if (inputs[0].getType() != ArrayType::DOUBLE ||
            inputs[0].getType() == ArrayType::COMPLEX_DOUBLE)
        {
            matlabPtr->feval(u"error", 0, 
                std::vector<Array>({ factory.createScalar("Input must be double array") }));
        }

        if (outputs.size() > 1) {
            matlabPtr->feval(u"error", 0, 
                std::vector<Array>({ factory.createScalar("Only one output is returned") }));
        }
    }
};

