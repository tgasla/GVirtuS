# Written By: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
#             Department of Applied Science
#             Raffaele Montella <raffaele.montella@uniparthenope.it>,
#             Department of Applied Science
#
# Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
#            School of Computer Science, University College Dublin

project(gvirtus-plugin-cufft)
find_package(CUDAToolkit REQUIRED)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

resolve_cuda_library_version(cufft CUFFT_VERSION)

gvirtus_add_backend(cufft ${CUFFT_VERSION}
        backend/CufftHandler.cpp)
target_link_libraries(${PROJECT_NAME}
    CUDA::cufft
)

gvirtus_add_frontend(cufft ${CUFFT_VERSION}
    frontend/Cufft.cpp
    frontend/CufftFrontend.cpp)
