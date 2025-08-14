# download_json.cmake
if(NOT DEFINED OUTPUT_FILE)
    message(FATAL_ERROR "OUTPUT_FILE not defined")
endif()

file(DOWNLOAD
    https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp
    ${OUTPUT_FILE}
    STATUS status
    LOG log
)

list(GET status 0 status_code)
if(NOT status_code EQUAL 0)
    message(FATAL_ERROR "Failed to download json.hpp. Status: ${status}. Log: ${log}")
endif()