# Peano toolchain for AIE2P tile compilation
# Not required for runtime, only for kernel JIT compilation

find_program(Peano_CXX aie2p-none-unknown-elf-g++)
find_program(Peano_AR aie2p-none-unknown-elf-ar)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Peano
    REQUIRED_VARS Peano_CXX Peano_AR
)
