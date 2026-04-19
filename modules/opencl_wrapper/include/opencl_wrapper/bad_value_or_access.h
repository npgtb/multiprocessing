#ifndef STATUSOR_BAD_VALUE_OR_ACCESS_H
#define STATUSOR_BAD_VALUE_OR_ACCESS_H

#include <exception>
#include <string>

namespace clw{
    class BadValueOrAccess : public std::exception{
        public:
            //Constructs a excepction with the given string contents
            explicit BadValueOrAccess(std::string msg):message_(msg){}

            //Returns message contents of the exception
            const char* what() const noexcept override{
                return message_.c_str();
            }

        private:
            std::string message_;
    };
}

#endif