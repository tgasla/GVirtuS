include_directories(include)

if(NOT ${PROJECT_NAME} STREQUAL "gvirtus-common")
    include_directories(${GVIRTUS_HOME}/include)
    link_directories(${GVIRTUS_HOME}/lib)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGVIRTUS_HOME=\\\"${GVIRTUS_HOME}\\\"")
set(CMAKE_SKIP_RPATH ON)

# Include the support to external projects
include(ExternalProject)

# Set the external install location
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/external)

# Include the log4cplus module (downloads or finds it)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/log4cplus.cmake)

find_package(Threads REQUIRED)

if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/gvirtus)
    message(STATUS "Installing ${CMAKE_CURRENT_SOURCE_DIR}/include/gvirtus to ${GVIRTUS_HOME}/include")
    install(DIRECTORY include/gvirtus DESTINATION ${GVIRTUS_HOME}/include)
endif()

macro(gvirtus_install_target target_name)
    install(TARGETS ${target_name}
        LIBRARY DESTINATION ${GVIRTUS_HOME}/lib
        ARCHIVE DESTINATION ${GVIRTUS_HOME}/lib
        RUNTIME DESTINATION ${GVIRTUS_HOME}/bin
        INCLUDES DESTINATION ${GVIRTUS_HOME}/include)
endmacro()

function(gvirtus_add_backend)
    if(ARGC LESS 3)
        message(FATAL_ERROR "Usage: gvirtus_add_backend(wrapped_library version source1 ...)")
    endif()
    list(GET ARGV 0 wrapped_library)
    list(REMOVE_AT ARGV 0)
    list(GET ARGV 0 version)
    list(REMOVE_AT ARGV 0)
    if(NOT "${PROJECT_NAME}" STREQUAL "gvirtus-plugin-${wrapped_library}")
        message(FATAL_ERROR "This project should be named gvirtus-plugin-${wrapped_library}")
    endif()
    add_library(${PROJECT_NAME} SHARED
            ${ARGV})
    target_link_libraries(${PROJECT_NAME} ${LIBLOG4CPLUS} gvirtus-common gvirtus-communicators)
    install(TARGETS ${PROJECT_NAME}
            LIBRARY DESTINATION ${GVIRTUS_HOME}/lib)
endfunction()

function(gvirtus_add_frontend)
    if(ARGC LESS 3)
        message(FATAL_ERROR "Usage: gvirtus_add_frontend(wrapped_library version source1 ...)")
    endif()
    list(GET ARGV 0 wrapped_library)
    list(REMOVE_AT ARGV 0)
    list(GET ARGV 0 version)
    list(REMOVE_AT ARGV 0)
    add_library(${wrapped_library} SHARED ${ARGV})
    target_link_libraries(${wrapped_library} ${LIBLOG4CPLUS} gvirtus-common gvirtus-communicators gvirtus-frontend)

    # Version properties
    set_target_properties(${wrapped_library} PROPERTIES VERSION ${version})
    string(REGEX REPLACE "\\..*" "" soversion ${version})
    set_target_properties(${wrapped_library} PROPERTIES SOVERSION ${soversion})

    # Generate and apply linker version script
    set(version_tag "lib${wrapped_library}.so.${soversion}")
    set(version_script_body "${version_tag} {\n global:\n*;\n};")
    set(version_script_path "${CMAKE_CURRENT_BINARY_DIR}/lib${wrapped_library}.ver")
    file(WRITE "${version_script_path}" "${version_script_body}")
    target_link_options(${wrapped_library} PRIVATE
        "-Wl,--version-script=${version_script_path}"
    )
    install(TARGETS ${wrapped_library}
        LIBRARY DESTINATION ${GVIRTUS_HOME}/lib/frontend)
endfunction()

# Version detection
function(resolve_cuda_library_version target out_var)
    find_package(CUDAToolkit REQUIRED)
    get_target_property(lib_path CUDA::${target} IMPORTED_LOCATION)
    set(version "UNKNOWN")
    if(lib_path)
        set(resolved "${lib_path}")
        while(IS_SYMLINK "${resolved}")
            execute_process(
            COMMAND readlink "${resolved}"
            OUTPUT_VARIABLE resolved
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        endwhile()
        string(REGEX REPLACE ".*\\.so\\.([0-9]+(\\.[0-9]+)*)" "\\1" version "${resolved}")
    endif()
    set(${out_var} "${version}" PARENT_SCOPE)
    message(STATUS "Resolved version for ${target}: ${version}")
endfunction()

function(maybe_add_cuda_plugin libname subdir)
    find_package(CUDAToolkit REQUIRED)
    if(TARGET CUDA::${libname})
        get_target_property(lib_location CUDA::${libname} IMPORTED_LOCATION)
        message(STATUS "CUDA library ${libname} found at ${lib_location}, adding ${subdir}")
        add_subdirectory(${subdir})
    else()
        message(WARNING "CUDA library ${libname} not found, skipping ${subdir}")
    endif()
endfunction()