# cmake_policy(SET CMP0104 NEW)

# # Set required CUDA architectures
# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
# endif()

# set(CMAKE_CXX_STANDARD 23)
# set(CMAKE_CXX_EXTENSIONS ON)
# set(CMAKE_CUDA_STANDARD 17)

cuda_add_executable(add_grid.e add_grid.cu OPTIONS --cudart=shared)
#set_target_properties(add_grid.e PROPERTIES CUDA_RUNTIME_LIBRARY shared EXTRA_NVCCFLAGS --cudart=shared)
install(TARGETS add_grid.e RUNTIME DESTINATION ${GVIRTUS_HOME}/demo/cudart)

cuda_add_executable(dummy0.e dummy0.cu OPTIONS --cudart=shared)
#set_target_properties(dummy0.e PROPERTIES CUDA_RUNTIME_LIBRARY shared EXTRA_NVCCFLAGS --cudart=shared)
install(TARGETS dummy0.e RUNTIME DESTINATION ${GVIRTUS_HOME}/demo/cudart)

cuda_add_executable(launchKernel.e launchKernel.cu OPTIONS --cudart=shared)
#set_target_properties(launchKernel.e PROPERTIES CUDA_RUNTIME_LIBRARY shared EXTRA_NVCCFLAGS --cudart=shared)
install(TARGETS launchKernel.e RUNTIME DESTINATION ${GVIRTUS_HOME}/demo/cudart)

message(STATUS "COMPILING...")
message(STATUS LD_LIBRARY_PATH ${LD_LIBRARY_PATH})
message(STATUS EXTRA_NVCC_FLAGS ${EXTRA_NVCC_FLAGS})
cuda_add_executable(matrixMul.e matrixMul.cu OPTIONS --cudart=shared)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set_target_properties(matrixMul.e PROPERTIES
        CUDA_RUNTIME_LIBRARY Shared
        EXTRA_NVCCFLAGS --cudart=shared)
install(TARGETS matrixMul.e RUNTIME DESTINATION ${GVIRTUS_HOME}/demo/cudart)