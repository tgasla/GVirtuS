project(gvirtus-plugin-cublas)
find_package(CUDAToolkit REQUIRED)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

resolve_cuda_library_version(cublas CUBLAS_VERSION)

gvirtus_add_backend(cublas ${CUBLAS_VERSION}
    backend/CublasHandler_Extension.cpp
    backend/CublasHandler_Helper.cpp
    backend/CublasHandler_Level1.cpp
    backend/CublasHandler_Level2.cpp
    backend/CublasHandler_Level3.cpp
    backend/CublasHandler_Lt.cpp
    backend/CublasHandler.cpp
)

target_link_libraries(${PROJECT_NAME}
    CUDA::cublas
    CUDA::cublasLt # maybe not needed
)

gvirtus_add_frontend(cublas ${CUBLAS_VERSION}
    frontend/Cublas_extension.cpp
    frontend/Cublas_helper.cpp
    frontend/Cublas_level1.cpp
    frontend/Cublas_level2.cpp
    frontend/Cublas_level3.cpp
    frontend/CublasFrontend.cpp
)

# This is for the cublasLt.so library
gvirtus_add_frontend(cublasLt ${CUBLAS_VERSION}
    frontend/CublasLt.cpp
    frontend/CublasFrontend.cpp
)