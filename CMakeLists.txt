cmake_minimum_required(VERSION 3.14)  # for add_link_options and implicit target directories.
project("vibeasr.cpp" C CXX)
include(CheckIncludeFileCXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# option list
option(BITNET_ARM_TL1    "bitnet.cpp: use tl1 on arm platform"    OFF)
option(BITNET_X86_TL2    "bitnet.cpp: use tl2 on x86 platform"    OFF)


set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)

# override ggml options
set(GGML_BITNET_ARM_TL1    ${BITNET_ARM_TL1})
set(GGML_BITNET_X86_TL2    ${BITNET_X86_TL2})

if (GGML_BITNET_ARM_TL1)
    add_compile_definitions(GGML_BITNET_ARM_TL1)
endif()
if (GGML_BITNET_X86_TL2)
    add_compile_definitions(GGML_BITNET_X86_TL2)
endif()

if (CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fpermissive)
endif()

find_package(Threads REQUIRED)

set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS  OFF CACHE BOOL "" FORCE)

# Portable binary, not -march=native. ggml's GGML_NATIVE default bakes the build
# host's full ISA into every translation unit -- measured consequence: a cloud VM
# migrated to a host without AVX512-FP16 mid-session and the binary died with SIGILL
# inside ggml_init's F16 table setup. The VibeASR kernels do not need native codegen:
# they carry their own AVX-512/VNNI paths behind target attributes with runtime
# dispatch, so a baseline-AVX2 build still runs the wide kernels on capable hosts
# and falls back cleanly on others. Override with -DGGML_NATIVE=ON for a
# single-machine build if you accept the portability loss.
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_AVX2   ON  CACHE BOOL "" FORCE)
set(GGML_FMA    ON  CACHE BOOL "" FORCE)
set(GGML_F16C   ON  CACHE BOOL "" FORCE)

add_subdirectory(src)
add_subdirectory(3rdparty/llama.cpp)

# Add standalone ASR inference executable
add_executable(asr_infer
    demo/asr_infer.cpp
    src/vae.cpp
    src/vae.h
)

target_include_directories(asr_infer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/ggml/include
)

target_link_libraries(asr_infer PRIVATE
    common
    llama
    ${CMAKE_THREAD_LIBS_INIT}
)

target_compile_features(asr_infer PRIVATE cxx_std_11)

set_target_properties(asr_infer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)


# Add streaming ASR server executable (persistent process with token-level streaming)
add_executable(asr_stream_server
    src/asr_server.cpp
    src/vae.cpp
    src/vae.h
)

target_include_directories(asr_stream_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/ggml/include
)

target_link_libraries(asr_stream_server PRIVATE
    common
    llama
    ${CMAKE_THREAD_LIBS_INIT}
)

target_compile_features(asr_stream_server PRIVATE cxx_std_11)

set_target_properties(asr_stream_server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    OUTPUT_NAME asr_stream_server
)


# Kernel correctness + throughput harness (bench/kernel_bench.cpp).
# Links ggml so it exercises the same kernel objects inference uses.
add_executable(kernel_bench bench/kernel_bench.cpp)

# The scalar references state exact mul-then-add semantics; letting gcc contract
# them to fma would make the "exact match" checks compare against a different
# rounding than the one the deterministic runtime paths guarantee.
set_source_files_properties(bench/kernel_bench.cpp PROPERTIES COMPILE_FLAGS "-ffp-contract=off")

target_include_directories(kernel_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/ggml/include
)

target_link_libraries(kernel_bench PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(kernel_bench PRIVATE cxx_std_11)

set_target_properties(kernel_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)


# Re-quantise one tensor of an already-quantised GGUF, leaving the I2_S body alone
# (tools/requant_lm_head.cpp). llama-quantize cannot do this.
add_executable(requant_lm_head tools/requant_lm_head.cpp)

target_include_directories(requant_lm_head PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/ggml/include
)

target_link_libraries(requant_lm_head PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(requant_lm_head PRIVATE cxx_std_11)

set_target_properties(requant_lm_head PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# install

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(LLAMA_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}
    CACHE PATH "Location of header files")
set(LLAMA_LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}
    CACHE PATH "Location of library files")
set(LLAMA_BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR}
    CACHE PATH "Location of binary files")
set(LLAMA_BUILD_NUMBER ${BUILD_NUMBER})
set(LLAMA_BUILD_COMMIT ${BUILD_COMMIT})
set(LLAMA_INSTALL_VERSION 0.0.${BUILD_NUMBER})

get_target_property(GGML_DIRECTORY ggml SOURCE_DIR)
get_directory_property(GGML_DIR_DEFINES DIRECTORY ${GGML_DIRECTORY} COMPILE_DEFINITIONS)
get_target_property(GGML_TARGET_DEFINES ggml COMPILE_DEFINITIONS)
set(GGML_TRANSIENT_DEFINES ${GGML_TARGET_DEFINES} ${GGML_DIR_DEFINES})
get_target_property(GGML_LINK_LIBRARIES ggml LINK_LIBRARIES)

get_directory_property(LLAMA_TRANSIENT_DEFINES COMPILE_DEFINITIONS)

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/LlamaConfigVersion.cmake
    VERSION ${LLAMA_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/LlamaConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/LlamaConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Llama)

set_target_properties(llama PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/llama.h)
install(TARGETS llama LIBRARY PUBLIC_HEADER)
