#include <iostream>

#include "backend_interface.hpp"

int main() {
  std::cout << "Data Engine go brr" << std::endl;
  BackendInterface backend;
  backend.initialize();
  return 0;
}
