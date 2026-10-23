cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
find_package(QtAV REQUIRED)
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
set(CMAKE_AUTOMOC ON)

add_executable(simpleplayer main.cpp playerwindow.cpp playerwindow.h)
target_include_directories(simpleplayer PRIVATE ${QTAVWIDGETS_INCLUDE_DIRS})
target_link_libraries(simpleplayer ${QTAVWIDGETS_LIBRARIES} Qt${QT_VERSION_MAJOR}::Widgets)