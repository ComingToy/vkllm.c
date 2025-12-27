#ifdef _SUPPORT_16BIT_STORAGE
#extension GL_EXT_shader_16bit_storage: require
#endif
#ifdef _SUPPORT_FLOAT16_ARITHMETIC
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

layout(local_size_x_id = 253, local_size_y_id = 254, local_size_z_id = 255) in;

struct ShapeConstant
{
    uint shapes[4];
    uint strides[4];
};

uint get_thread_id()
{
    uint tid = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_WorkGroupSize.x +
                 gl_GlobalInvocationID.z * gl_WorkGroupSize.x * gl_WorkGroupSize.y;
	return tid;
}
