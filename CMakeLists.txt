cmake_minimum_required(VERSION 2.4)

# OpenCV Config
find_package(OpenCV REQUIRED)
message("-- OPENCV include:   " ${OpenCV_INCLUDE_DIRS})
message("-- OPENCV libs dir:  " ${OpenCV_LIB_DIR})
message("-- OPENCV libs:   " ${OpenCV_LIBS} )

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
LINK_DIRECTORIES(${OpenCV_LIB_DIR} ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
ADD_DEFINITIONS(-DDEBUG -ggdb -O3 -Wall -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -fbranch-probabilities)

if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

################################


ADD_EXECUTABLE(
  lk_serial src/lk_serial.cpp
)
ADD_EXECUTABLE(
  lk_parallel src/lk_parallel.cpp
)
ADD_EXECUTABLE(
  sub_serial src/sub_serial.cpp
)
ADD_EXECUTABLE(
  sub_parallel src/sub_parallel.cpp
)

TARGET_LINK_LIBRARIES(
  sub_parallel
  ${OPENCV_LIBRARIES}
  ${OpenCV_LIBS}
)
TARGET_LINK_LIBRARIES(
  lk_parallel
  ${OPENCV_LIBRARIES}
  ${OpenCV_LIBS}
)
TARGET_LINK_LIBRARIES(
  lk_serial
  ${OPENCV_LIBRARIES}
  ${OpenCV_LIBS}
)
TARGET_LINK_LIBRARIES(
  sub_serial
  ${OPENCV_LIBRARIES}
  ${OpenCV_LIBS}
)


################################

foreach(dir ${DIRS})
  FILE(GLOB files ${dir}/*.h*)
  LIST(APPEND h_files ${files})
endforeach(dir)
add_custom_target(TAGS etags --members -l c++ ${h_files} ${SRCS})

