cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0104 NEW)

project(GVirtuS LANGUAGES CXX)

# ===== LZ4 check & link =====
find_package(LZ4 QUIET)
if(LZ4_FOUND)
    link_libraries(LZ4::LZ4)
else()
    find_library(LZ4_LIB lz4 REQUIRED)
    link_libraries(${LZ4_LIB})
endif()

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

# ===== gvirtus-common =====
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

gvirtus_install_target(gvirtus-common)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/nlohmann DESTINATION ${GVIRTUS_HOME}/include)
if (LOG4CPLUS_EXTERNALLY_DOWNLOADED)
    add_dependencies(gvirtus-common log4cplus)
    install(DIRECTORY ${EXTERNAL_INSTALL_LOCATION}/include/log4cplus DESTINATION ${GVIRTUS_HOME}/include)
    install(DIRECTORY ${EXTERNAL_INSTALL_LOCATION}/lib DESTINATION ${GVIRTUS_HOME})
endif()
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/etc DESTINATION ${GVIRTUS_HOME})

# ===== gvirtus-communicators =====
add_library(gvirtus-communicators SHARED
    src/communicators/Buffer.cpp
    src/communicators/CommunicatorFactory.cpp
    src/communicators/Endpoint_Tcp.cpp
    src/communicators/Endpoint_Rdma.cpp
    src/communicators/Endpoint_Hybrid.cpp
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
add_library(gvirtus-communicators-ib SHARED
    src/communicators/rdma/ktmrdma.cpp
    src/communicators/rdma/RdmaCommunicator.cpp
)
target_link_libraries(gvirtus-communicators-ib gvirtus-communicators rdmacm ibverbs)
gvirtus_install_target(gvirtus-communicators-ib)

## HYBRID COMMUNICATOR
add_library(gvirtus-communicators-hybrid SHARED
    src/communicators/hybrid/HybridCommunicator.cpp
)
target_include_directories(gvirtus-communicators-hybrid
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(gvirtus-communicators-hybrid
    gvirtus-communicators
    gvirtus-communicators-tcp
    gvirtus-communicators-ib
)
gvirtus_install_target(gvirtus-communicators-hybrid)

# ===== FRONTEND =====
add_library(gvirtus-frontend SHARED
    src/frontend/Frontend.cpp
)
target_include_directories(gvirtus-frontend
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(gvirtus-frontend
    PRIVATE
        gvirtus-communicators
        gvirtus-communicators-hybrid
        Threads::Threads
        lz4
)
gvirtus_install_target(gvirtus-frontend)

# ===== BACKEND =====
add_executable(gvirtus-backend
    src/backend/Backend.cpp
    src/backend/main.cpp
    src/backend/Process.cpp
    src/backend/Property.cpp
)
target_include_directories(gvirtus-backend
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(gvirtus-backend
    PRIVATE
        gvirtus-communicators
        gvirtus-communicators-hybrid
        Threads::Threads
        rdmacm
        ibverbs
)
gvirtus_install_target(gvirtus-backend)

# ===== Plugins =====
add_subdirectory(plugins/cudart)
add_subdirectory(plugins/cudadr)
add_subdirectory(plugins/cudnn) 

maybe_add_cuda_plugin(cublas plugins/cublas)
maybe_add_cuda_plugin(cufft plugins/cufft)
maybe_add_cuda_plugin(cusolver plugins/cusolver)
maybe_add_cuda_plugin(cusparse plugins/cusparse)
maybe_add_cuda_plugin(curand plugins/curand)
maybe_add_cuda_plugin(nvrtc plugins/nvrtc)

add_subdirectory(plugins/nvml)

# ===== Tests =====
add_subdirectory(tests)

# add_subdirectory(tools/protocol-generator)
