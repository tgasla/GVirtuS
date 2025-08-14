# log4cplus
find_path(LOG4CPLUS_INCLUDE_DIR log4cplus/logger.h)
find_library(LOG4CPLUS_LIBRARY log4cplus)

if(NOT LOG4CPLUS_INCLUDE_DIR OR NOT LOG4CPLUS_LIBRARY)
    message(STATUS "log4cplus not found. Downloading and installing it.")
    include(ExternalProject)

    set(LOG4CPLUS_INSTALL_DIR ${EXTERNAL_INSTALL_LOCATION})
    set(LOG4CPLUS_INCLUDE_DIR ${LOG4CPLUS_INSTALL_DIR}/include)
    set(LOG4CPLUS_LIBRARY ${LOG4CPLUS_INSTALL_DIR}/lib/liblog4cplus.so)

    ExternalProject_Add(log4cplus
        URL https://downloads.sourceforge.net/log4cplus/log4cplus-stable/2.1.2/log4cplus-2.1.2.tar.gz
        TIMEOUT 360
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/log4cplus-prefix
        BUILD_IN_SOURCE 1
        CONFIGURE_COMMAND ./configure --prefix=${LOG4CPLUS_INSTALL_DIR} CFLAGS=-fPIC CPPFLAGS=-I${LOG4CPLUS_INCLUDE_DIR} LDFLAGS=-L${LOG4CPLUS_INSTALL_DIR}/lib
        BUILD_COMMAND make
        INSTALL_COMMAND make install
    )
    set(LOG4CPLUS_EXTERNALLY_DOWNLOADED TRUE)
else()
    message(STATUS "log4cplus found in ${LOG4CPLUS_INCLUDE_DIR}")
    set(LOG4CPLUS_EXTERNALLY_DOWNLOADED FALSE)
endif()

# Export variables
set(LIBLOG4CPLUS ${LOG4CPLUS_LIBRARY})
include_directories(${LOG4CPLUS_INCLUDE_DIR})