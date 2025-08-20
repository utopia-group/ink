cmake_minimum_required(VERSION 3.29)

# Set the project name
project(Evaluation)

# Read environment variables
set(TIMEOUT "$ENV{timeout}" CACHE STRING "Timeout in seconds")
set(INPUT_DIR "$ENV{input_dir}" CACHE STRING "Input directory path")
set(OUTPUT_DIR "$ENV{output_dir}" CACHE STRING "Output directory path")
set(EXECUTABLE "$ENV{executable}" CACHE STRING "Executable path")

if(NOT DEFINED ENV{SRC_EXT})
    message(FATAL_ERROR "SRC_EXT, the extension of the source files, is not defined")
endif()
set(SRC_EXT "$ENV{SRC_EXT}")

if(NOT DEFINED ENV{OUT_EXT})
    message(FATAL_ERROR "OUT_EXT, the extension of the output files, is not defined")
endif()
set(OUT_EXT "$ENV{OUT_EXT}")

add_custom_target(run_eval)
file(GLOB input_files "${INPUT_DIR}/*.${SRC_EXT}")
foreach(input_file ${input_files})
  get_filename_component(input_file_name ${input_file} NAME)

  # Remove the known extension (e.g., ".txt") from the end of the filename
  string(LENGTH ".${SRC_EXT}" ext_len)
  string(LENGTH "${input_file_name}" name_len)
  math(EXPR base_len "${name_len} - ${ext_len}")
  string(SUBSTRING "${input_file_name}" 0 ${base_len} input_file_name_without_ext)
  set(output_file "${OUTPUT_DIR}/${input_file_name_without_ext}.${OUT_EXT}")

  add_custom_command(
    COMMAND ${EXECUTABLE} ${input_file} ${output_file}
    DEPENDS ${input_file}
    OUTPUT ${output_file}
    WORKING_DIRECTORY ${ROOT_DIR}
  )

  add_custom_target(
    ${input_file_name}
    DEPENDS ${output_file}
  )

  add_dependencies(run_eval ${input_file_name})
endforeach()
