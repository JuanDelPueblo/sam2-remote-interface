# Add library for backend interface
add_library(backend_interface backend_interface.cpp)
target_link_libraries(backend_interface PRIVATE CURL::libcurl nlohmann_json::nlohmann_json)

# Add executable
add_executable(DataEngine main.cpp)
target_link_libraries(DataEngine PRIVATE backend_interface CURL::libcurl nlohmann_json::nlohmann_json)