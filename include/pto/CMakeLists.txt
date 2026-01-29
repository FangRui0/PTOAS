add_subdirectory(IR)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Transforms)
  add_subdirectory(Transforms)
endif()
