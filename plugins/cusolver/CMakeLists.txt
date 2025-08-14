# Written By: Raffaele Montella <raffaele.montella@uniparthenope.it>,
#              Department of Science and Technologies
#
# Edited By: Theodoros Aslanidis <theodoros.aslanidis@uniparthenope.it>,
#              School of Computer Science, University College Dublin

project(gvirtus-plugin-cusolver)
find_package(CUDAToolkit REQUIRED)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

resolve_cuda_library_version(cusolver CUSOLVER_VERSION)

gvirtus_add_backend(cusolver ${CUSOLVER_VERSION}
    backend/CusolverHandler.cpp
)

target_link_libraries(${PROJECT_NAME}
    CUDA::cusolver
)

gvirtus_add_frontend(cusolver ${CUSOLVER_VERSION}
    frontend/Cusolver.cpp
    frontend/CusolverFrontend.cpp
    frontend/Cusolver_helper.cpp
)
