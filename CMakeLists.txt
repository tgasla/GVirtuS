cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0104 NEW)

project(GVirtuS LANGUAGES CXX)
enable_testing()

# Set required CUDA architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if("$ENV{GVIRTUS_HOME}" STREQUAL "")
    message(STATUS "Setting GVIRTUS_HOME=$ENV{HOME}/GVirtuS")
    set(GVIRTUS_HOME "$ENV{HOME}/GVirtuS")
    set($ENV{GVIRTUS_HOME} "${GVIRTUS_HOME}")
else()
    message(STATUS "Using GVIRTUS_HOME=$ENV{GVIRTUS_HOME}")
    set(GVIRTUS_HOME "$ENV{GVIRTUS_HOME}")
endif()

include(cmake/GVirtuS.cmake)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/nlohmann/json.hpp
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/include/nlohmann
    COMMAND ${CMAKE_COMMAND} -DOUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/include/nlohmann/json.hpp -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/download_json.cmake
)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_library(gvirtus-common SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/include/nlohmann/json.hpp
    src/common/Decoder.cpp
    src/common/Encoder.cpp
    src/common/JSON.cpp
    src/common/LD_Lib.cpp
    src/common/MessageDispatcher.cpp
    src/common/Mutex.cpp
    src/common/Observable.cpp
    src/common/Observer.cpp
    src/common/SignalException.cpp
    src/common/SignalState.cpp
    src/common/Util.cpp
)

target_link_libraries(gvirtus-common stdc++fs ${CMAKE_DL_LIBS} ${LIBLOG4CPLUS} rdmacm ibverbs)

##target_include_directories(gvirtus-common PRIVATE /usr/include/infiniband)
##target_include_directories(gvirtus-common PRIVATE /usr/include/rdma)

gvirtus_install_target(gvirtus-common)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/nlohmann DESTINATION ${GVIRTUS_HOME}/include)
if (LOG4CPLUS_EXTERNALLY_DOWNLOADED)
    add_dependencies(gvirtus-common log4cplus)
    install(DIRECTORY ${EXTERNAL_INSTALL_LOCATION}/include/log4cplus DESTINATION ${GVIRTUS_HOME}/include)
    install(DIRECTORY ${EXTERNAL_INSTALL_LOCATION}/lib DESTINATION ${GVIRTUS_HOME})
endif()
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/etc DESTINATION ${GVIRTUS_HOME})

add_library(gvirtus-communicators SHARED
    src/communicators/Buffer.cpp
    src/communicators/CommunicatorFactory.cpp
    src/communicators/Endpoint_Tcp.cpp
    src/communicators/Endpoint_Rdma.cpp
    src/communicators/EndpointFactory.cpp
    src/communicators/rdma/ktmrdma.cpp
    src/communicators/Result.cpp
)
target_link_libraries(gvirtus-communicators gvirtus-common rdmacm ibverbs)
gvirtus_install_target(gvirtus-communicators)

## TCP COMMUNICATOR
add_library(gvirtus-communicators-tcp SHARED
    src/communicators/tcp/TcpCommunicator.cpp)
target_link_libraries(gvirtus-communicators-tcp gvirtus-communicators)
gvirtus_install_target(gvirtus-communicators-tcp)

## IB COMMUNICATOR
## TODO: PROPERLY SETUP CMAKE AND PROJECT TO USE RDMA
add_library(gvirtus-communicators-ib SHARED
    src/communicators/rdma/ktmrdma.cpp
    src/communicators/rdma/RdmaCommunicator.cpp
)
##target_include_directories(gvirtus-communicators-ib PUBLIC /usr/include/infiniband /usr/include/rdma /src/communicators/rdma)
target_link_libraries(gvirtus-communicators-ib gvirtus-communicators rdmacm ibverbs)
gvirtus_install_target(gvirtus-communicators-ib)

add_library(gvirtus-frontend SHARED
    src/frontend/Frontend.cpp)
target_link_libraries(gvirtus-frontend gvirtus-communicators Threads::Threads lz4)
gvirtus_install_target(gvirtus-frontend)

add_executable(gvirtus-backend
    src/backend/Backend.cpp
    src/backend/main.cpp
    src/backend/Process.cpp
    src/backend/Property.cpp)
target_link_libraries(gvirtus-backend gvirtus-communicators Threads::Threads rdmacm ibverbs)
gvirtus_install_target(gvirtus-backend)

add_subdirectory(plugins/cudart) # Assuming this plugin is always available
add_subdirectory(plugins/cudadr) # CUDA is not part of CUDA toolkit. Internally, we check if cuda is available

maybe_add_cuda_plugin(cublas plugins/cublas) # part of CUDA toolkit
maybe_add_cuda_plugin(cufft plugins/cufft) # part of CUDA toolkit
maybe_add_cuda_plugin(cusolver plugins/cusolver) # part of CUDA toolkit
maybe_add_cuda_plugin(cusparse plugins/cusparse) # part of CUDA toolkit
maybe_add_cuda_plugin(curand plugins/curand) # part of CUDA toolkit
maybe_add_cuda_plugin(nvrtc plugins/nvrtc) # part of CUDA toolkit
maybe_add_cuda_plugin(nvml plugins/nvml) # typically bundled with the NVIDIA GPU driver

add_subdirectory(plugins/cudnn) # CUDNN is not part of CUDA toolkit. Internally, we check if cudnn is available

add_subdirectory(tests)

#add_subdirectory(tools/protocol-generator)