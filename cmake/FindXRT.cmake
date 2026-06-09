# Find XRT library for AMD NPU backend
# Sets: XRT_FOUND, XRT_INCLUDE_DIRS, XRT_LIBRARIES

find_path(XRT_INCLUDE_DIRS
    NAMES xrt/xrt_bo.h xrt.h
    PATHS
        ${CMAKE_SOURCE_DIR}/third_party/xrt-dev/usr/include
        /opt/xilinx/xrt/include
        /opt/xilinx/include
        /usr/include
    PATH_SUFFIXES
        xrt
)

set(_XRT_LIB_SEARCH_PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/lib
    /opt/xilinx/xrt/lib
    /opt/xilinx/lib
    /opt/xilinx/lib/x86_64-linux-gnu
)

# Runtime .so.2 from libxrt2; unversioned .so symlinks come from libxrt-dev
set(_XRT_SAVE_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES .so.2 .so)

foreach(_xrt_lib IN ITEMS XRT_COREUTIL:xrt_coreutil XRT_CORE:xrt_core XRT_CXX:xrt++)
    string(REPLACE ":" ";" _parts ${_xrt_lib})
    list(GET _parts 0 _var)
    list(GET _parts 1 _name)
    find_library(${_var}_LIB
        NAMES ${_name}
        PATHS /usr/lib/x86_64-linux-gnu ${_XRT_LIB_SEARCH_PATHS}
    )
endforeach()

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_XRT_SAVE_SUFFIXES})

set(XRT_LIBRARIES "")
if(XRT_COREUTIL_LIB)
    list(APPEND XRT_LIBRARIES ${XRT_COREUTIL_LIB})
endif()
if(XRT_CORE_LIB)
    list(APPEND XRT_LIBRARIES ${XRT_CORE_LIB})
endif()
if(XRT_CXX_LIB)
    list(APPEND XRT_LIBRARIES ${XRT_CXX_LIB})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XRT
    REQUIRED_VARS XRT_INCLUDE_DIRS XRT_LIBRARIES
)

find_path(UUID_INCLUDE_DIRS uuid/uuid.h
    PATHS
        ${CMAKE_SOURCE_DIR}/third_party/uuid-dev/usr/include
        /usr/include
)

find_library(UUID_LIBRARY NAMES uuid
    PATHS
        /usr/lib/x86_64-linux-gnu
        ${CMAKE_SOURCE_DIR}/third_party/uuid-dev/usr/lib/x86_64-linux-gnu
)

if(XRT_FOUND AND NOT TARGET XRT::XRT)
    set(_XRT_INTERFACE_INCLUDES "${XRT_INCLUDE_DIRS}")
    if(UUID_INCLUDE_DIRS)
        list(APPEND _XRT_INTERFACE_INCLUDES "${UUID_INCLUDE_DIRS}")
    endif()

    set(_XRT_INTERFACE_LIBS ${XRT_LIBRARIES})
    if(UUID_LIBRARY)
        list(APPEND _XRT_INTERFACE_LIBS ${UUID_LIBRARY})
    endif()

    add_library(XRT::XRT INTERFACE IMPORTED)
    set_target_properties(XRT::XRT PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_XRT_INTERFACE_INCLUDES}"
        INTERFACE_LINK_LIBRARIES "${_XRT_INTERFACE_LIBS}"
    )
endif()
