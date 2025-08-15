#pragma once

#include <stdexcept>
#include <string>

namespace gvirtus::common {

class SignalException : public std::runtime_error {
 public:
  explicit SignalException(const std::string& message) : std::runtime_error(message) {}
};

}
