cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(QTAV)

set(QTAV_MAJOR 1)
set(QTAV_MINOR 13)
set(QTAV_PATCH 0)
set(PROJECT_VERSION ${QTAV_MAJOR}.${QTAV_MINOR}.${QTAV_PATCH})
set(SO_VERSION ${QTAV_MAJOR})

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(POLICY CMP0063) # visibility. since 3.3
  cmake_policy(SET CMP0063 NEW)
endif()
if(POLICY CMP0071)
  cmake_policy(SET CMP0071 NEW)
endif()
set(CMAKE_CXX_VISIBILITY_PRESET hidden) #use with -fdata-sections -ffunction-sections to reduce target size
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)


option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_PLAYERS "Build players" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_QT5OPENGL "Build with Qt5 OpenGL module" ON)
option(BUILD_QML "Build QML interfaces" ON)
option(BUILD_Vulkan "Build Vulkan Renderer" ON)

list(APPEND CMAKE_FIND_ROOT_PATH ${QTDIR})

# Qt major version: prefer Qt6, fall back to Qt5. Force one with -DQT_VERSION_MAJOR=5|6
if(NOT QT_VERSION_MAJOR)
  find_package(QT NAMES Qt6 Qt5 COMPONENTS Core REQUIRED)
endif()
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Gui)
set(QT_VERSION ${Qt${QT_VERSION_MAJOR}Core_VERSION})
macro(qtav_add_resources)
  if(QT_VERSION_MAJOR EQUAL 6)
    qt6_add_resources(${ARGN})
  else()
    qt5_add_resources(${ARGN})
  endif()
endmacro()
set(CMAKE_CXX_STANDARD 17) # Qt6 requires it, and some QtAV sources already use C++17 features

if(QT_VERSION_MAJOR EQUAL 6)
  # no qmake in cmake-only Qt6 builds
  set(QT_INSTALL_PREFIX ${QT6_INSTALL_PREFIX})
  set(QT_INSTALL_HEADERS ${QT_INSTALL_PREFIX}/include)
  set(QT_INSTALL_LIBS ${QT_INSTALL_PREFIX}/lib)
  set(QT_INSTALL_BINS ${QT_INSTALL_PREFIX}/bin)
  set(QT_INSTALL_QML ${QT_INSTALL_PREFIX}/qml)
else()
  get_target_property(qmake Qt5::qmake LOCATION)
  foreach(QT_DIR HEADERS LIBS BINS QML PREFIX)
    execute_process(
      COMMAND ${qmake} -query QT_INSTALL_${QT_DIR}
      OUTPUT_VARIABLE QT_INSTALL_${QT_DIR}
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  endforeach()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${QT5_CFLAGS}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(POLICY CMP0063) # visibility. since 3.3
  cmake_policy(SET CMP0063 NEW)
endif()
set(CMAKE_CXX_VISIBILITY_PRESET hidden) #use with -fdata-sections -ffunction-sections to reduce target size
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON) # for .moc
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
if(NOT CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(QTAV_INSTALL_HEADERS ${CMAKE_INSTALL_PREFIX}/include)
  set(QTAV_INSTALL_LIBS ${CMAKE_INSTALL_PREFIX}/lib)
  set(QTAV_INSTALL_BINS ${CMAKE_INSTALL_PREFIX}/bin)
  set(QTAV_INSTALL_QML ${QTAV_INSTALL_LIBS}/qml)
else()
  set(CMAKE_INSTALL_PREFIX ${QT_INSTALL_PREFIX} CACHE PATH "default install path" FORCE)
  set(QTAV_INSTALL_HEADERS ${QT_INSTALL_HEADERS})
  set(QTAV_INSTALL_LIBS ${QT_INSTALL_LIBS})
  set(QTAV_INSTALL_BINS ${QT_INSTALL_BINS})
  set(QTAV_INSTALL_QML ${QT_INSTALL_QML})
endif()

message(STATUS "Qt version: ${QT_VERSION}")
message(STATUS "Qt prefix: ${QT_INSTALL_PREFIX}")
message(STATUS "QtAV headers prefix: ${QTAV_INSTALL_HEADERS}")
message(STATUS "QtAV libs prefix: ${QTAV_INSTALL_LIBS}")
message(STATUS "QtAV bins prefix: ${QTAV_INSTALL_BINS}")
message(STATUS "QtAV qml prefix: ${QTAV_INSTALL_QML}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

#set(CMAKE_DEBUG_POSTFIX d)

install(FILES
  cmake/FindQtAV.cmake
  DESTINATION ${QTAV_INSTALL_LIBS}/cmake
)

if(CMAKE_GENERATOR MATCHES "Visual Studio.*")
  set(UCRT_INCLUDE $ENV{UniversalCRTSDKDir}Include\\$ENV{UCRTVersion}\\ucrt)
  set(VC_ARCH $ENV{Platform})
  if(NOT VC_ARCH)
    set(VC_ARCH x86)
  endif()
  set(UCRT_LIB $ENV{UniversalCRTSDKDir}Lib\\$ENV{UCRTVersion}\\ucrt\\${VC_ARCH})
  if(EXISTS ${UCRT_INCLUDE})
  message("VS project. ucrt: ${UCRT_LIB}")
    include_directories(${UCRT_INCLUDE})
    link_libraries(${CMAKE_LIBRARY_PATH_FLAG}${UCRT_LIB})
  endif()
endif()

add_subdirectory(src)

find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets)
if(Qt${QT_VERSION_MAJOR}Widgets_FOUND)
  add_subdirectory(widgets)
endif()

if(BUILD_QML)
  find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Qml Quick)
  if(Qt${QT_VERSION_MAJOR}Qml_FOUND AND Qt${QT_VERSION_MAJOR}Quick_FOUND)
      add_subdirectory(qml)
  endif()
endif()

if(BUILD_EXAMPLES OR BUILD_PLAYERS)
  add_subdirectory(examples)
endif()


