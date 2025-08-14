# Written By: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
#             Department of Applied Science
#
# Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
#             School of Computer Science, University College Dublin

project(gvirtus-plugin-cudnn)

function(configure_cudnn_plugin)
    find_package(CUDA REQUIRED)

    find_path(CUDNN_INCLUDE_DIRECTORY
        cudnn.h
        PATHS ${CUDA_INCLUDE_DIRS})

    if(NOT CUDNN_INCLUDE_DIRECTORY)
        message(WARNING "cudnn.h not found, skipping plugins/cudnn")
        return()
    endif()

    message(STATUS "cuDNN include directory found at: ${CUDNN_INCLUDE_DIRECTORY}")

    include_directories(${CUDNN_INCLUDE_DIRECTORY} ${CUDA_INCLUDE_DIRS})

    get_filename_component(CUDA_LIBRARIES_PATH ${CUDA_CUDART_LIBRARY} DIRECTORY FALSE)

    find_library(CUDNN_LIBRARY
        cudnn
        PATHS ${CUDA_LIBRARIES_PATH})

    if(NOT CUDNN_LIBRARY)
        message(WARNING "cuDNN library not found, skipping plugins/cudnn")
        return()
    endif()

    message(STATUS "cuDNN library found at ${CUDNN_LIBRARY}, adding plugins/cudnn")

    set(CUDNN_VERSION "${CUDNN_LIBRARY}")
    while(IS_SYMLINK "${CUDNN_VERSION}")
        execute_process(COMMAND readlink "${CUDNN_VERSION}"
            OUTPUT_VARIABLE CUDNN_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endwhile()
    string(REGEX REPLACE ".*\\.so\\.([0-9]+(\\.[0-9]+)*)" "\\1" CUDNN_VERSION "${CUDNN_VERSION}")

    message(STATUS "Resolved version for cuDNN: ${CUDNN_VERSION}")

    gvirtus_add_backend(cudnn ${CUDNN_VERSION}
        backend/CudnnHandler.cpp)

    target_link_libraries(${PROJECT_NAME} ${CUDNN_LIBRARY} ${CUDA_CUDART_LIBRARY})

    gvirtus_add_frontend(cudnn ${CUDNN_VERSION}
        frontend/Cudnn.cpp
        frontend/Cudnn_helper.cpp
        frontend/CudnnFrontend.cpp)

endfunction()

# Call the function to configure cudnn
configure_cudnn_plugin()