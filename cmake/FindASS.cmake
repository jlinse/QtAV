#.rst:
# FindASS
# -------
# Finds the ASS library
#
# This will define the following variables::
#
# ASS_FOUND - system has ASS
# ASS_INCLUDE_DIRS - the ASS include directory
# ASS_LIBRARIES - the ASS libraries
#
# and the following imported targets::
#
#   ASS::ASS   - The ASS library

find_package(PkgConfig)

if(PKG_CONFIG_FOUND)
  pkg_check_modules(PC_ASS libass QUIET)
endif()

include(FindPackageHandleStandardArgs)
include(SelectLibraryConfigurations)
include(CMakeFindDependencyMacro)


set(SEARCH_PATH "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")

find_path(ASS_INCLUDE_DIR NAMES ass/ass.h PATHS ${SEARCH_PATH}/include NO_DEFAULT_PATH)

find_library(ASS_LIBRARY_RELEASE NAMES ass libass PATHS ${SEARCH_PATH}/lib/ NO_DEFAULT_PATH)
find_library(ASS_LIBRARY_DEBUG NAMES ass libass PATHS ${SEARCH_PATH}/debug/lib/ NO_DEFAULT_PATH)

SET(ASS_LIBRARY
  debug ${ASS_LIBRARY_DEBUG}
  optimized ${ASS_LIBRARY_RELEASE}
)
if(ASS_LIBRARY)
  set(ASS_VERSION ${PC_ASS_VERSION})

  find_package(Freetype)
  find_package(harfbuzz CONFIG)

  find_library(ASS_DEPEND_FRIBIDI_LIBRARY_DEBUG NAMES fribidi PATHS ${SEARCH_PATH}/lib/ NO_DEFAULT_PATH)
  find_library(ASS_DEPEND_FRIBIDI_LIBRARY_RELEASE NAMES fribidi PATHS ${SEARCH_PATH}/debug/lib/ NO_DEFAULT_PATH)
  SET(ASS_DEPEND_FRIBIDI_LIBRARY
    debug ${ASS_DEPEND_FRIBIDI_LIBRARY_DEBUG}
    optimized ${ASS_DEPEND_FRIBIDI_LIBRARY_RELEASE}
  )

  set(ASS_FOUND TRUE)

  if(NOT TARGET ASS::ASS)
    add_library(ASS::ASS INTERFACE IMPORTED)
    target_link_libraries(ASS::ASS INTERFACE ${ASS_LIBRARY} Freetype::Freetype harfbuzz::harfbuzz ${ASS_DEPEND_FRIBIDI_LIBRARY})
    target_include_directories(ASS::ASS INTERFACE ${ASS_INCLUDE_DIR})
  endif()
endif()

find_package_handle_standard_args(ASS
REQUIRED_VARS 

ASS_LIBRARY 
ASS_INCLUDE_DIR
ASS_DEPEND_FRIBIDI_LIBRARY
Freetype_FOUND
harfbuzz_FOUND

VERSION_VAR ASS_VERSION)
