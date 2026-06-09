# Find XRT library for AMD NPU backend
# Sets: XRT_INCLUDE_DIRS, XRT_LIBRARIES

find_path(XRT_INCLUDE_DIRS xrt.h
    PATHS
        /opt/xilinx/include
        /opt/xilinx/include/xrt
        /usr/include
        /usr/include/xrt
    PATH_SUFFIXES
        xrt
)

find_library(XRT_LIBRARIES NAMES xrt_coreutil xrt_core xilinxopencl
    PATHS
        /opt/xilinx/lib
        /opt/xilinx/lib/x86_64-linux-gnu
        /usr/lib
        /usr/lib/x86_64-linux-gnu
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XRT
    REQUIRED_VARS XRT_LIBRARIES XRT_INCLUDE_DIRS
)

if(XRT_FOUND AND NOT TARGET XRT::XRT)
    add_library(XRT::XRT INTERFACE IMPORTED)
    set_target_properties(XRT::XRT PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${XRT_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${XRT_LIBRARIES}"
    )
endif()
