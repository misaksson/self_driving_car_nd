#ifndef ANSI_COLOR_CODES_H
#define ANSI_COLOR_CODES_H

#include <string>

namespace ANSI {
  const std::string BLACK = "\033[;30m";
  const std::string RED = "\033[;31m";
  const std::string GREEN = "\033[;32m";
  const std::string YELLOW = "\033[;33m";
  const std::string BLUE = "\033[;34m";
  const std::string PURPLE = "\033[;35m";
  const std::string CYAN = "\033[;36m";
  const std::string WHITE = "\033[;37m";

  const std::string RESET = "\033[0m";
}

#endif /* ANSI_COLOR_CODES_H */
