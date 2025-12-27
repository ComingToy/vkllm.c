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
    uvec3 glb_dims = gl_WorkGroupSize * gl_NumWorkGroups;
    uint glb_tid = gl_GlobalInvocationID.z * glb_dims.y * glb_dims.x + gl_GlobalInvocationID.y * glb_dims.x + gl_GlobalInvocationID.x;
	return glb_tid;
}
