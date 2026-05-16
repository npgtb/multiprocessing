#ifndef OPENCL_WRAPPER_VALUE_OR_H
#define OPENCL_WRAPPER_VALUE_OR_H

#include <concepts>
#include <variant>
#include <type_traits>
#include <opencl_wrapper/opencl.h>
#include <opencl_wrapper/bad_value_or_access.h>

namespace clw{
    //Absl inspired handling for the error/value situations
    template <typename T>
    class ErrorOr{
        public:

            static_assert(!std::is_same_v<T, cl_int>, "ErrorOr<T> can't hold a cl_int");

            //Default to a cl error
            ErrorOr():data_(CL_INVALID_VALUE){}

            //Handle receiving error codes
            ErrorOr(cl_int error):data_(error){}

            //Handle receiving values
            template <typename U> requires std::constructible_from<T, U>
            ErrorOr(U&& value):data_(T(std::forward<U>(value))){}

            //OK to access value
            bool ok() const noexcept{
                return std::holds_alternative<T>(data_);
            }
            
            //Retrieve the error code
            cl_int error() const{
                if(!ok()){
                    return std::get<cl_int>(data_);
                }
                throw BadValueOrAccess("ErrorOr object does not hold a OpenCL error code!");
            }

            //Retrieve data. left value calls only return a reference to the data
            const T& value() const &{
                if(ok()){
                    return std::get<T>(data_);
                }
                std::string error_msg = "ErrorOr object does not hold a value!";
                throw BadValueOrAccess(error_msg);
            }
            T& value() &{
                if(ok()){
                    return std::get<T>(data_);
                }
                std::string error_msg = "ErrorOr object does not hold a value!";
                throw BadValueOrAccess(error_msg);
            }

            //Retrieve data. right value calls hands over the ownership of the data, std::move(..).value()
            T&& value() &&{
                if(ok()){
                    return std::move(std::get<T>(data_));
                }
                throw BadValueOrAccess("ErrorOr object does not hold a value!");
            }

        private:
            std::variant<T, cl_int> data_;
    };
}

#endif