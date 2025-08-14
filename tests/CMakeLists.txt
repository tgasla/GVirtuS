# Written By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
#             School of Computer Science, University College Dublin

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)

# Find GoogleTest package (make sure it is installed or handled otherwise)
find_package(GTest QUIET)
if (NOT GTest_FOUND)
    message(STATUS "GTest not found, fetching with FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
    )
    FetchContent_MakeAvailable(googletest)
    # After FetchContent_MakeAvailable, GTest targets become available, 
    # but variables like GTest_INCLUDE_DIRS might not be set, so you may want to set them manually:
    set(GTEST_INCLUDE_DIRS ${googletest_SOURCE_DIR}/include)
    set(GTEST_LIBRARIES gtest gtest_main)
endif()

# Add your CUDA source files manually
set(TEST_SOURCES
    test_cudart.cu
    test_nvrtc.cu
    test_curand.cu
    test_cublas.cu
    test_cufft.cu
    test_cudnn.cu
    test_cudadr.cu
    test_cusolver.cu
    test_cusparse.cu
)

## Alternatively, you can use file(GLOB ...) to automatically find all .cu files in a directory
# file(GLOB TEST_SOURCES "*.cu")
# But this way you cannot control the order of tests, so if a test crashes the backend, all the others will be failed too.

# Loop over source files and create separate targets
foreach(TEST_SRC ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SRC})
    
    set_target_properties(${TEST_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_include_directories(${TEST_NAME} PRIVATE
        ${GTEST_INCLUDE_DIRS}
    )

    target_link_libraries(${TEST_NAME} PRIVATE
        GTest::GTest
        GTest::Main
        cuda # This is cudaDr library
        curand
        cudart
        cublas
        cublasLt
        cufft
        cudnn
        cusolver
        cusparse
        nvrtc
    )

    # Register the test with ctest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()