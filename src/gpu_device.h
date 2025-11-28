#ifndef __VKLLM_GPU_DEVICE_H__
#define __VKLLM_GPU_DEVICE_H__

struct gpu_device
{
	int id;
};


extern struct gpu_device* new_gpu_device(int id);
#endif
