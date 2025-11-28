#include "gpu_device.h"
#include <stdlib.h>

struct gpu_device* new_gpu_device(int id)
{
	struct gpu_device* dev = (struct gpu_device*)malloc(sizeof(struct gpu_device));	

	if (!dev){
		return NULL;
	}

	return NULL;
}
