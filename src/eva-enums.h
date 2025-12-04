#pragma once
#include <cstdint>

#define DEFINE_OPERATOR_OR(enum_type) \
inline constexpr enum_type operator|(enum_type lhs, enum_type rhs)       \
{ \
    return (enum_type) ((uint32_t)lhs | (uint32_t)rhs); \
}

#define DEFINE_OPERATOR_OR_ASSIGN(enum_type) \
inline enum_type& operator|=(enum_type& lhs, enum_type rhs)                \
{ \
    lhs = lhs | rhs; \
    return lhs; \
}

#define DEFINE_HAS_FLAGS(enum_type) \
inline constexpr bool hasFlag(enum_type value, enum_type flag)        \
{ \
    return (((uint32_t)value & (uint32_t)flag) == (uint32_t)flag); \
}


namespace eva {


enum Result : int32_t {
    SUCCESS                                 = 0,
    NOT_READY                               = 1,
    TIMEOUT                                 = 2,
    EVENT_SET                               = 3,
    EVENT_RESET                             = 4,
    INCOMPLETE                              = 5,
    ERROR_OUT_OF_HOST_MEMORY                = -1,
    ERROR_OUT_OF_DEVICE_MEMORY              = -2,
    ERROR_INITIALIZATION_FAILED             = -3,
    ERROR_DEVICE_LOST                       = -4,
    ERROR_MEMORY_MAP_FAILED                 = -5,
    ERROR_LAYER_NOT_PRESENT                 = -6,
    ERROR_EXTENSION_NOT_PRESENT             = -7,
    ERROR_FEATURE_NOT_PRESENT               = -8,
    ERROR_INCOMPATIBLE_DRIVER               = -9,
    ERROR_TOO_MANY_OBJECTS                  = -10,
    ERROR_FORMAT_NOT_SUPPORTED              = -11,
    ERROR_FRAGMENTED_POOL                   = -12,
    ERROR_UNKNOWN                           = -13,
    ERROR_OUT_OF_POOL_MEMORY                = -1000069000,
    ERROR_INVALID_EXTERNAL_HANDLE           = -1000072003,
    ERROR_FRAGMENTATION                     = -1000161000,
    ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS    = -1000257000,
    PIPELINE_COMPILE_REQUIRED               = 1000297000,
    ERROR_SURFACE_LOST                      = -1000000000,
    ERROR_NATIVE_WINDOW_IN_USE              = -1000000001,
    SUBOPTIMAL                              = 1000001003,
    ERROR_OUT_OF_DATE                       = -1000001004,
};


enum class IMAGE_LAYOUT : uint32_t {
    UNDEFINED                 = 0,
    GENERAL                   = 1,
    COLOR_ATTACHMENT          = 2,
    DEPTH_STENCIL_ATTACHMENT  = 3,
    DEPTH_STENCIL_READ_ONLY   = 4,
    SHADER_READ_ONLY          = 5,
    TRANSFER_SRC              = 6,
    TRANSFER_DST              = 7,
    PREINITIALIZED            = 8,
    PRESENT_SRC               = 1000001002,
    MAX_ENUM                  = 0x7FFFFFFF,
};


enum class FORMAT : uint32_t {
    UNDEFINED = 0,
    R4G4_UNORM_PACK8 = 1,
    R4G4B4A4_UNORM_PACK16 = 2,
    B4G4R4A4_UNORM_PACK16 = 3,
    R5G6B5_UNORM_PACK16 = 4,
    B5G6R5_UNORM_PACK16 = 5,
    R5G5B5A1_UNORM_PACK16 = 6,
    B5G5R5A1_UNORM_PACK16 = 7,
    A1R5G5B5_UNORM_PACK16 = 8,
    R8_UNORM = 9,
    R8_SNORM = 10,
    R8_UINT = 13,
    R8_SINT = 14,
    R8_SRGB = 15,
    R8G8_UNORM = 16,
    R8G8_SNORM = 17,
    R8G8_UINT = 20,
    R8G8_SINT = 21,
    R8G8_SRGB = 22,
    R8G8B8_UNORM = 23,
    R8G8B8_SNORM = 24,
    R8G8B8_UINT = 27,
    R8G8B8_SINT = 28,
    R8G8B8_SRGB = 29,
    B8G8R8_UNORM = 30,
    B8G8R8_SNORM = 31,
    B8G8R8_UINT = 34,
    B8G8R8_SINT = 35,
    B8G8R8_SRGB = 36,
    R8G8B8A8_UNORM = 37,
    R8G8B8A8_SNORM = 38,
    R8G8B8A8_UINT = 41,
    R8G8B8A8_SINT = 42,
    R8G8B8A8_SRGB = 43,
    B8G8R8A8_UNORM = 44,
    B8G8R8A8_SNORM = 45,
    B8G8R8A8_UINT = 48,
    B8G8R8A8_SINT = 49,
    B8G8R8A8_SRGB = 50,
    A8B8G8R8_UNORM_PACK32 = 51,
    A8B8G8R8_SNORM_PACK32 = 52,
    A8B8G8R8_UINT_PACK32 = 55,
    A8B8G8R8_SINT_PACK32 = 56,
    A8B8G8R8_SRGB_PACK32 = 57,
    A2R10G10B10_UNORM_PACK32 = 58,
    A2R10G10B10_UINT_PACK32 = 62,
    A2B10G10R10_UNORM_PACK32 = 64,
    A2B10G10R10_UINT_PACK32 = 68,
    R16_UNORM = 70,
    R16_SNORM = 71,
    R16_UINT = 74,
    R16_SINT = 75,
    R16_SFLOAT = 76,
    R16G16_UNORM = 77,
    R16G16_SNORM = 78,
    R16G16_UINT = 81,
    R16G16_SINT = 82,
    R16G16_SFLOAT = 83,
    R16G16B16_UNORM = 84,
    R16G16B16_SNORM = 85,
    R16G16B16_UINT = 88,
    R16G16B16_SINT = 89,
    R16G16B16_SFLOAT = 90,
    R16G16B16A16_UNORM = 91,
    R16G16B16A16_SNORM = 92,
    R16G16B16A16_UINT = 95,
    R16G16B16A16_SINT = 96,
    R16G16B16A16_SFLOAT = 97,
    R32_UINT = 98,
    R32_SINT = 99,
    R32_SFLOAT = 100,
    R32G32_UINT = 101,
    R32G32_SINT = 102,
    R32G32_SFLOAT = 103,
    R32G32B32_UINT = 104,
    R32G32B32_SINT = 105,
    R32G32B32_SFLOAT = 106,
    R32G32B32A32_UINT = 107,
    R32G32B32A32_SINT = 108,
    R32G32B32A32_SFLOAT = 109,
    R64_UINT = 110,
    R64_SINT = 111,
    R64_SFLOAT = 112,
    R64G64_UINT = 113,
    R64G64_SINT = 114,
    R64G64_SFLOAT = 115,
    R64G64B64_UINT = 116,
    R64G64B64_SINT = 117,
    R64G64B64_SFLOAT = 118,
    R64G64B64A64_UINT = 119,
    R64G64B64A64_SINT = 120,
    R64G64B64A64_SFLOAT = 121,
    B10G11R11_UFLOAT_PACK32 = 122,
    E5B9G9R9_UFLOAT_PACK32 = 123,
    D16_UNORM = 124,
    X8_D24_UNORM_PACK32 = 125,
    D32_SFLOAT = 126,
    S8_UINT = 127,
    D16_UNORM_S8_UINT = 128,
    D24_UNORM_S8_UINT = 129,
    D32_SFLOAT_S8_UINT = 130,
    BC1_RGB_UNORM_BLOCK = 131,
    BC1_RGB_SRGB_BLOCK = 132,
    BC1_RGBA_UNORM_BLOCK = 133,
    BC1_RGBA_SRGB_BLOCK = 134,
    BC2_UNORM_BLOCK = 135,
    BC2_SRGB_BLOCK = 136,
    BC3_UNORM_BLOCK = 137,
    BC3_SRGB_BLOCK = 138,
    BC4_UNORM_BLOCK = 139,
    BC4_SNORM_BLOCK = 140,
    BC5_UNORM_BLOCK = 141,
    BC5_SNORM_BLOCK = 142,
    BC6H_UFLOAT_BLOCK = 143,
    BC6H_SFLOAT_BLOCK = 144,
    BC7_UNORM_BLOCK = 145,
    BC7_SRGB_BLOCK = 146,
    MAX_ENUM = 0x7FFFFFFF
};


enum class IMAGE_TILING : uint32_t {
    OPTIMAL = 0,
    LINEAR = 1,
    MAX_ENUM = 0x7FFFFFFF,
};


enum class IMAGE_TYPE : uint32_t {
    _1D = 0,
    _2D = 1,
    _3D = 2,
    MAX_ENUM = 0x7FFFFFFF,
};


enum class COMPONENT_SWIZZLE : uint32_t {
    IDENTITY = 0,
    ZERO = 1,
    ONE = 2,
    R = 3,
    G = 4,
    B = 5,
    A = 6,
    MAX_ENUM = 0x7FFFFFFF,
};
DEFINE_OPERATOR_OR(COMPONENT_SWIZZLE)


enum class IMAGE_VIEW_TYPE : uint32_t {
    _1D = 0,
    _2D = 1,
    _3D = 2,
    CUBE = 3,
    _1D_ARRAY = 4,
    _2D_ARRAY = 5,
    CUBE_ARRAY = 6,
    MAX_ENUM = 0x7FFFFFFF,
};


enum class COMPARE_OP : uint32_t {
    NEVER = 0,
    LESS = 1,
    EQUAL = 2,
    LESS_OR_EQUAL = 3,
    GREATER = 4,
    NOT_EQUAL = 5,
    GREATER_OR_EQUAL = 6,
    ALWAYS = 7,
    MAX_ENUM = 0x7FFFFFFF,
};


enum class BORDER_COLOR : uint32_t {
    FLOAT_TRANSPARENT_BLACK = 0,
    INT_TRANSPARENT_BLACK = 1,
    FLOAT_OPAQUE_BLACK = 2,
    INT_OPAQUE_BLACK = 3,
    FLOAT_OPAQUE_WHITE = 4,
    INT_OPAQUE_WHITE = 5,
    MAX_ENUM = 0x7FFFFFFF,
};


enum class FILTER : uint32_t {
    NEAREST = 0,
    LINEAR = 1,
    CUBIC = 1000015000,
    MAX_ENUM = 0x7FFFFFFF,
};


enum class SAMPLER_ADDRESS_MODE : uint32_t {
    REPEAT = 0,
    MIRRORED_REPEAT = 1,
    CLAMP_TO_EDGE = 2,
    CLAMP_TO_BORDER = 3,
    MIRROR_CLAMP_TO_EDGE = 4,
    MAX_ENUM = 0x7FFFFFFF,
};


enum class SAMPLER_MIPMAP_MODE : uint32_t {
    NEAREST = 0,
    LINEAR = 1,
    MAX_ENUM = 0x7FFFFFFF,
};


enum class DESCRIPTOR_TYPE : uint32_t {
    SAMPLER                         = 0,
    COMBINED_IMAGE_SAMPLER          = 1,
    SAMPLED_IMAGE                   = 2,
    STORAGE_IMAGE                   = 3,
    UNIFORM_TEXEL_BUFFER            = 4,
    STORAGE_TEXEL_BUFFER            = 5,
    UNIFORM_BUFFER                  = 6,
    STORAGE_BUFFER                  = 7,
    UNIFORM_BUFFER_DYNAMIC          = 8,
    STORAGE_BUFFER_DYNAMIC          = 9,
    INPUT_ATTACHMENT                = 10,
    INLINE_UNIFORM_BLOCK            = 1000138000,
    ACCELERATION_STRUCTURE          = 1000150000,
    MAX_ENUM                        = 0x7FFFFFFF,
};


enum class PIPELINE_BIND_POINT : uint32_t {
    GRAPHICS = 0,
    COMPUTE = 1,
    RAY_TRACING = 1000165000,
    MAX_ENUM = 0x7FFFFFFF,
};


enum class ACCESS : uint64_t {
    NONE                                  =              0ULL,
    INDIRECT_COMMAND_READ                 =     0x00000001ULL,
    INDEX_READ                            =     0x00000002ULL,
    VERTEX_ATTRIBUTE_READ                 =     0x00000004ULL,
    UNIFORM_READ                          =     0x00000008ULL,
    INPUT_ATTACHMENT_READ                 =     0x00000010ULL,
    SHADER_READ                           =     0x00000020ULL,
    SHADER_WRITE                          =     0x00000040ULL,
    COLOR_ATTACHMENT_READ                 =     0x00000080ULL,
    COLOR_ATTACHMENT_WRITE                =     0x00000100ULL,
    DEPTH_STENCIL_ATTACHMENT_READ         =     0x00000200ULL,
    DEPTH_STENCIL_ATTACHMENT_WRITE        =     0x00000400ULL,
    TRANSFER_READ                         =     0x00000800ULL,
    TRANSFER_WRITE                        =     0x00001000ULL,
    HOST_READ                             =     0x00002000ULL,
    HOST_WRITE                            =     0x00004000ULL,
    MEMORY_READ                           =     0x00008000ULL,
    MEMORY_WRITE                          =     0x00010000ULL,
    COMMAND_PREPROCESS_READ               =     0x00020000ULL,
    COMMAND_PREPROCESS_WRITE              =     0x00040000ULL,
    COLOR_ATTACHMENT_READ_NONCOHERENT     =     0x00080000ULL,
    CONDITIONAL_RENDERING_READ            =     0x00100000ULL,
    ACCELERATION_STRUCTURE_READ           =     0x00200000ULL,
    ACCELERATION_STRUCTURE_WRITE          =     0x00400000ULL,
    FRAGMENT_SHADING_RATE_ATTACHMENT_READ =     0x00800000ULL,
    FRAGMENT_DENSITY_MAP_READ             =     0x01000000ULL,
    TRANSFORM_FEEDBACK_WRITE              =     0x02000000ULL,
    TRANSFORM_FEEDBACK_COUNTER_READ       =     0x04000000ULL,
    TRANSFORM_FEEDBACK_COUNTER_WRITE      =     0x08000000ULL,
    SHADER_SAMPLED_READ                   =    0x100000000ULL,
    SHADER_STORAGE_READ                   =    0x200000000ULL,
    SHADER_STORAGE_WRITE                  =    0x400000000ULL,
    VIDEO_DECODE_READ                     =    0x800000000ULL,
    VIDEO_DECODE_WRITE                    =   0x1000000000ULL,
    VIDEO_ENCODE_READ                     =   0x2000000000ULL,
    VIDEO_ENCODE_WRITE                    =   0x4000000000ULL,
    SHADER_BINDING_TABLE_READ             =  0x10000000000ULL,
    DESCRIPTOR_BUFFER_READ                =  0x20000000000ULL,
    OPTICAL_FLOW_READ                     =  0x40000000000ULL,
    OPTICAL_FLOW_WRITE                    =  0x80000000000ULL,
    MICROMAP_READ                         = 0x100000000000ULL,
    MICROMAP_WRITE                        = 0x200000000000ULL,
};
DEFINE_OPERATOR_OR(ACCESS)
DEFINE_HAS_FLAGS(ACCESS)


enum class IMAGE_CREATE : uint32_t {
    NONE                                = 0x00000000,
    SPARSE_BINDING                      = 0x00000001,
    SPARSE_RESIDENCY                    = 0x00000002,
    SPARSE_ALIASED                      = 0x00000004,
    MUTABLE_FORMAT                      = 0x00000008,
    CUBE_COMPATIBLE                     = 0x00000010,
    SPLIT_INSTANCE_BIND_REGIONS         = 0x00000040,
    BLOCK_TEXEL_VIEW_COMPATIBLE         = 0x00000080,
    EXTENDED_USAGE                      = 0x00000100,
    DISJOINT                            = 0x00000200,
    ALIAS                               = 0x00000400,
    PROTECTED                           = 0x00000800,
    SAMPLE_LOCATIONS_COMPATIBLE_DEPTH   = 0x00001000,
    SUBSAMPLED                          = 0x00004000,
};
DEFINE_OPERATOR_OR(IMAGE_CREATE)
DEFINE_HAS_FLAGS(IMAGE_CREATE)


enum class IMAGE_USAGE : uint32_t {
    TRANSFER_SRC                        = 0x00000001,
    TRANSFER_DST                        = 0x00000002,
    SAMPLED                             = 0x00000004,
    STORAGE                             = 0x00000008,
    COLOR_ATTACHMENT                    = 0x00000010,
    DEPTH_STENCIL_ATTACHMENT            = 0x00000020,
    TRANSIENT_ATTACHMENT                = 0x00000040,
    INPUT_ATTACHMENT                    = 0x00000080,
    FRAGMENT_SHADING_RATE_ATTACHMENT    = 0x00000100,
    HOST_TRANSFER                       = 0x00400000,
};
DEFINE_OPERATOR_OR(IMAGE_USAGE)
DEFINE_HAS_FLAGS(IMAGE_USAGE)


enum class MEMORY_PROPERTY : uint32_t {
    DEVICE_LOCAL        = 0x00000001,
    HOST_VISIBLE       = 0x00000002,
    HOST_COHERENT      = 0x00000004,
    HOST_CACHED        = 0x00000008,
    LAZILY_ALLOCATED   = 0x00000010,
    PROTECTED          = 0x00000020,    
};
DEFINE_OPERATOR_OR(MEMORY_PROPERTY)
DEFINE_HAS_FLAGS(MEMORY_PROPERTY)


enum class QUEUE : uint32_t {
    GRAPHICS    = 0x00000001,
    COMPUTE     = 0x00000002,
    TRANSFER    = 0x00000004,
    MAX_ENUM    = 0x7FFFFFFF,
};
DEFINE_OPERATOR_OR(QUEUE)
DEFINE_HAS_FLAGS(QUEUE)


enum class PIPELINE_STAGE : uint64_t {
    NONE                              =              0ULL,
    TOP_OF_PIPE                       =     0x00000001ULL,
    DRAW_INDIRECT                     =     0x00000002ULL,
    VERTEX_INPUT                      =     0x00000004ULL,
    VERTEX_SHADER                     =     0x00000008ULL,
    TESSELLATION_CONTROL_SHADER       =     0x00000010ULL,
    TESSELLATION_EVALUATION_SHADER    =     0x00000020ULL,
    GEOMETRY_SHADER                   =     0x00000040ULL,
    FRAGMENT_SHADER                   =     0x00000080ULL,
    EARLY_FRAGMENT_TESTS              =     0x00000100ULL,
    LATE_FRAGMENT_TESTS               =     0x00000200ULL,
    COLOR_ATTACHMENT_OUTPUT           =     0x00000400ULL,
    COMPUTE_SHADER                    =     0x00000800ULL,
    TRANSFER                          =     0x00001000ULL,
    BOTTOM_OF_PIPE                    =     0x00002000ULL,
    HOST                              =     0x00004000ULL,
    ALL_GRAPHICS                      =     0x00008000ULL,
    ALL_COMMANDS                      =     0x00010000ULL,
    COMMAND_PREPROCESS                =     0x00020000ULL,
    CONDITIONAL_RENDERING             =     0x00040000ULL,
    TASK_SHADER                       =     0x00080000ULL,
    MESH_SHADER                       =     0x00100000ULL,
    RAY_TRACING_SHADER                =     0x00200000ULL,
    FRAGMENT_SHADING_RATE_ATTACHMENT  =     0x00400000ULL,
    FRAGMENT_DENSITY_PROCESS          =     0x00800000ULL,
    TRANSFORM_FEEDBACK                =     0x01000000ULL,
    ACCELERATION_STRUCTURE_BUILD      =     0x02000000ULL,
    VIDEO_DECODE                      =     0x04000000ULL,
    VIDEO_ENCODE                      =     0x08000000ULL,
    ACCELERATION_STRUCTURE_COPY       =     0x10000000ULL,
    OPTICAL_FLOW                      =     0x20000000ULL,
    MICROMAP_BUILD                    =     0x40000000ULL,
    COPY                              =    0x100000000ULL,
    RESOLVE                           =    0x200000000ULL,
    BLIT                              =    0x400000000ULL,
    CLEAR                             =    0x800000000ULL,
    INDEX_INPUT                       =   0x1000000000ULL,
    VERTEX_ATTRIBUTE_INPUT            =   0x2000000000ULL,
    PRE_RASTERIZATION_SHADERS         =   0x4000000000ULL,
    CONVERT_COOPERATIVE_VECTOR_MATRIX = 0x100000000000ULL,
};
DEFINE_OPERATOR_OR(PIPELINE_STAGE)
DEFINE_HAS_FLAGS(PIPELINE_STAGE)


enum class BUFFER_USAGE : uint32_t {
    TRANSFER_SRC                                    = 0x00000001,
    TRANSFER_DST                                    = 0x00000002,
    UNIFORM_TEXEL_BUFFER                            = 0x00000004,
    STORAGE_TEXEL_BUFFER                            = 0x00000008,
    UNIFORM_BUFFER                                  = 0x00000010,
    STORAGE_BUFFER                                  = 0x00000020,
    INDEX_BUFFER                                    = 0x00000040,
    VERTEX_BUFFER                                   = 0x00000080,
    INDIRECT_BUFFER                                 = 0x00000100,
    CONDITIONAL_RENDERING                           = 0x00000200,
    SHADER_BINDING_TABLE                            = 0x00000400,
    TRANSFORM_FEEDBACK_BUFFER                       = 0x00000800,
    TRANSFORM_FEEDBACK_COUNTER_BUFFER               = 0x00001000,
    SHADER_DEVICE_ADDRESS                           = 0x00020000,
    ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY    = 0x00080000,
    ACCELERATION_STRUCTURE_STORAGE                  = 0x00100000,
};
DEFINE_OPERATOR_OR(BUFFER_USAGE)
DEFINE_HAS_FLAGS(BUFFER_USAGE)


enum class SHADER_STAGE : uint32_t {
    NONE                      = 0x00000000,
    VERTEX                    = 0x00000001,
    TESSELLATION_CONTROL      = 0x00000002,
    TESSELLATION_EVALUATION   = 0x00000004,
    GEOMETRY                  = 0x00000008,
    FRAGMENT                  = 0x00000010,
    ALL_GRAPHICS              = 0x0000001F,
    COMPUTE                   = 0x00000020,
    TASK                      = 0x00000040,
    MESH                      = 0x00000080,
    RAYGEN                    = 0x00000100,
    ANY_HIT                   = 0x00000200,
    CLOSEST_HIT               = 0x00000400,
    MISS                      = 0x00000800,
    INTERSECTION              = 0x00001000,
    CALLABLE                  = 0x00002000,
    ALL                       = 0x7FFFFFFF,
};
DEFINE_OPERATOR_OR(SHADER_STAGE)
DEFINE_OPERATOR_OR_ASSIGN(SHADER_STAGE)
DEFINE_HAS_FLAGS(SHADER_STAGE)


enum class DESCRIPTOR_POOL_CREATE : uint32_t {
    NONE                        = 0,
    FREE_DESCRIPTOR_SET         = 0x00000001,
    UPDATE_AFTER_BIND           = 0x00000002,
    HOST_ONLY                   = 0x00000004,
};
DEFINE_OPERATOR_OR(DESCRIPTOR_POOL_CREATE)
DEFINE_HAS_FLAGS(DESCRIPTOR_POOL_CREATE)


enum class COMMAND_POOL_CREATE : uint32_t {
    NONE                        = 0,
    TRANSIENT                   = 0x00000001,
    RESET_COMMAND_BUFFER        = 0x00000002,
    PROTECTED                   = 0x00000004,
};
DEFINE_OPERATOR_OR(COMMAND_POOL_CREATE)
DEFINE_HAS_FLAGS(COMMAND_POOL_CREATE)


enum class COMMAND_BUFFER_USAGE : uint32_t {
    NONE                        = 0,
    ONE_TIME_SUBMIT             = 0x00000001,
    RENDER_PASS_CONTINUE        = 0x00000002,
    SIMULTANEOUS_USE            = 0x00000004,
};
DEFINE_OPERATOR_OR(COMMAND_BUFFER_USAGE)
DEFINE_HAS_FLAGS(COMMAND_BUFFER_USAGE)


enum class PRESENT_MODE : uint32_t {
    IMMEDIATE               = 0,
    MAILBOX                 = 1,
    FIFO                    = 2,
    FIFO_RELAXED            = 3,
    SHARED_DEMAND_REFRESH   = 1000111000,
    SHARED_CONTINUOUS_REFRESH = 1000111001,
    MAX_ENUM                = 0x7FFFFFFF,
};


enum class COLOR_SPACE : uint32_t {
    SRGB_NONLINEAR          = 0,
    DISPLAY_P3_NONLINEAR    = 1000104001,
    EXTENDED_SRGB_LINEAR    = 1000104002,
    DCI_P3_LINEAR           = 1000104003,
    DCI_P3_NONLINEAR        = 1000104004,
    BT709_LINEAR            = 1000104005,
    BT709_NONLINEAR         = 1000104006,
    BT2020_LINEAR           = 1000104007,
    HDR10_ST2084            = 1000104008,
    DOLBYVISION             = 1000104009,
    HDR10_HLG               = 1000104010,
    MAX_ENUM                = 0x7FFFFFFF,
};


enum class GEOMETRY_TYPE : uint32_t {
    TRIANGLES   = 0,
    AABBS       = 1,
    INSTANCES   = 2,
    MAX_ENUM    = 0x7FFFFFFF,
};


enum class ACCELERATION_STRUCTURE_TYPE : uint32_t {
    TOP_LEVEL = 0,
    BOTTOM_LEVEL = 1,
    GENERIC_KHR = 2,
};


enum class GEOMETRY : uint32_t {
    NONE    = 0,
    OPAQUE  = 0x00000001,
    NO_DUPLICATE_ANY_HIT_INVOCATION  = 0x00000002,
};
DEFINE_OPERATOR_OR(GEOMETRY)
DEFINE_HAS_FLAGS(GEOMETRY)


enum class GEOMETRY_INSTANCE : uint32_t {
    NONE                            = 0,
    TRIANGLE_FACING_CULL_DISABLE    = 0x00000001,
    TRIANGLE_FLIP_FACING            = 0x00000002,
    FORCE_OPAQUE                    = 0x00000004,
    FORCE_NO_OPAQUE                 = 0x00000008,
};
DEFINE_OPERATOR_OR(GEOMETRY_INSTANCE)
DEFINE_HAS_FLAGS(GEOMETRY_INSTANCE)


enum class BUILD_ACCELERATION_STRUCTURE : uint32_t {
    NONE = 0,
    ALLOW_UPDATE = 0x00000001,
    ALLOW_COMPACTION = 0x00000002,
    PREFER_FAST_TRACE = 0x00000004,
    PREFER_FAST_BUILD = 0x00000008,
    LOW_MEMORY = 0x00000010,
};
DEFINE_OPERATOR_OR(BUILD_ACCELERATION_STRUCTURE)
DEFINE_HAS_FLAGS(BUILD_ACCELERATION_STRUCTURE)



} // namespace eva