# vkllm.c

A high-performance Vulkan-based Large Language Model inference library written in C.

## Features

- **Vulkan Compute**: Leverages GPU compute shaders for fast inference
- **LLaMA 2 Support**: Complete implementation with GGUF model loading
- **Core Operators**: MatMul, Embedding, RMSNorm, Softmax, RoPE, FFN, and more
- **Computational Graph**: Efficient graph-based execution with automatic dependency management
- **FP16/FP32 Support**: Flexible precision for different use cases
- **Cross-Platform**: Works on Linux and macOS

## Prerequisites

### System Requirements
- **Vulkan SDK**: Version 1.2 or higher
- **Bazel**: Version 7.x (build system)
- **C Compiler**: GCC or Clang with C11 support
- **Python**: 3.13 (for tooling)

### Platform-Specific Requirements

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install vulkan-tools libvulkan-dev vulkan-validationlayers-dev spirv-tools
```

**macOS:**
```bash
brew install vulkan-tools vulkan-headers vulkan-loader spirv-tools
```

## Building

### 1. Install Bazel
Follow instructions at [bazel.build](https://bazel.build/install) for your platform.

### 2. Clone the Repository
```bash
git clone https://github.com/yourusername/vkllm.c.git
cd vkllm.c
```

### 3. Build the Library and Tests
```bash
# Build all targets
bazel build //...

# Or build specific components
bazel build //src/core:vkllm_core
bazel build //src/models:vkllm_models
```

### 4. Run Tests
```bash
# Run all tests
./run_all_tests.sh

# Or run individual tests
bazel test //tests:vkllm_test_op_matmul
bazel test //tests:vkllm_test_op_embedding
```

## Quick Start

### Basic Usage Pattern

Here's a minimal example showing how to use vkllm.c for inference:

```c
#include "src/core/vkllm_context.h"
#include "src/core/vkllm_tensor.h"
#include "src/core/vkllm_graph.h"
#include "src/models/vkllm_models_llama2.h"

int main() {
    // 1. Create a Vulkan context
    struct vkllm_context *context = NULL;
    vkllm_err_t err = vkllm_context_new(0, &context);  // 0 = first GPU device
    if (err != VKLLM_ERR_OK) {
        fprintf(stderr, "Failed to create context: %s\n", vkllm_err_s(err));
        return -1;
    }

    // 2. Load a LLaMA 2 model from GGUF
    struct vkllm_models_llama2 model;
    err = vkllm_models_llama2_load(context, &model, "path/to/model.gguf");
    if (err != VKLLM_ERR_OK) {
        fprintf(stderr, "Failed to load model: %s\n", vkllm_err_s(err));
        vkllm_context_free(context);
        return -1;
    }

    // 3. Tokenize input text
    struct vkllm_array_token_id *token_ids;
    err = vkllm_models_llama2_tokenize(&model, "Hello, world!", &token_ids);
    
    // 4. Create input tensor
    struct vkllm_tensor *input_toks = NULL;
    uint32_t input_shapes[] = {1, 1, 1, token_ids->used_n};
    vkllm_tensor_new(context, "input_toks", input_shapes, 
                     vkllm_dtype_uint32, VKLLM_OP_NONE, 
                     NULL, 0, NULL, 0, true, &input_toks);

    // 5. Build computational graph
    struct vkllm_graph *graph = NULL;
    vkllm_graph_new(context, &graph);
    vkllm_models_llama2_build_graph(context, &model, input_toks, graph);
    vkllm_graph_init(context, graph);

    // 6. Upload input data
    vkllm_commands_upload(context, graph->commands, input_toks,
                         (const uint8_t *)token_ids->data,
                         sizeof(uint32_t) * token_ids->used_n);

    // 7. Execute inference
    vkllm_graph_run(context, graph);
    vkllm_graph_post_run(context, graph);

    // 8. Access output (logits)
    struct vkllm_tensor *output = graph->output_node;
    // Process output logits...

    // 9. Cleanup
    vkllm_graph_free(context, graph);
    vkllm_tensor_free(context, input_toks);
    vkllm_array_token_id_free(token_ids);
    vkllm_models_llama2_free(context, &model);
    vkllm_context_free(context);
    
    return 0;
}
```

### Building Your Application

Create a BUILD file for your application:

```python
cc_binary(
    name = "my_inference_app",
    srcs = ["main.c"],
    deps = [
        "//src/core:vkllm_core",
        "//src/models:vkllm_models",
    ],
)
```

Build and run:
```bash
bazel build //:my_inference_app
./bazel-bin/my_inference_app
```

## API Overview

### Core Components

#### 1. Context (`vkllm_context`)
The main entry point that manages Vulkan device and pipelines.

```c
struct vkllm_context *context;
vkllm_context_new(device_id, &context);
vkllm_context_free(context);
```

#### 2. Tensor (`vkllm_tensor`)
4D tensor representation with GPU memory management.

```c
struct vkllm_tensor *tensor;
uint32_t shapes[] = {batch, channels, height, width};
vkllm_tensor_new(context, "name", shapes, vkllm_dtype_float16,
                 VKLLM_OP_NONE, NULL, 0, NULL, 0, true, &tensor);
vkllm_tensor_free(context, tensor);
```

#### 3. Graph (`vkllm_graph`)
Computational graph for managing tensor operations and execution order.

```c
struct vkllm_graph *graph;
vkllm_graph_new(context, &graph);
vkllm_graph_add_input(context, graph, input_tensor);
vkllm_graph_set_output(context, graph, output_tensor);
vkllm_graph_init(context, graph);
vkllm_graph_run(context, graph);
vkllm_graph_post_run(context, graph);
vkllm_graph_free(context, graph);
```

### Supported Operations

| Operation | Description | Header |
|-----------|-------------|--------|
| `VKLLM_OP_MATMUL` | Matrix multiplication with broadcasting | `vkllm_op_matmul.h` |
| `VKLLM_OP_EMBEDDING` | Token embedding lookup | `vkllm_op_embedding.h` |
| `VKLLM_OP_RMSNORM` | Root mean square normalization | `vkllm_op_rmsnorm.h` |
| `VKLLM_OP_SOFTMAX` | Softmax activation | `vkllm_op_softmax.h` |
| `VKLLM_OP_ROPE` | Rotary position embeddings | `vkllm_op_rope.h` |
| `VKLLM_OP_FFN_UP_AND_GATE` | Feed-forward with gating | `vkllm_op_ffn_up_and_gate.h` |
| `VKLLM_OP_BIN` | Binary operations (add, mul, etc.) | `vkllm_op_bin.h` |
| `VKLLM_OP_COPY` | Tensor copy/transfer | `vkllm_op_copy.h` |

### Data Types

```c
typedef enum {
    vkllm_dtype_float32,  // 32-bit float
    vkllm_dtype_float16,  // 16-bit float
    vkllm_dtype_int8,     // 8-bit integer
    vkllm_dtype_uint32,   // 32-bit unsigned integer
} vkllm_dtype_t;
```

## Examples

### Custom Model Implementation

You can build custom models by composing operations:

```c
// Create input tensor
struct vkllm_tensor *input;
uint32_t in_shapes[] = {1, 1, seq_len, hidden_dim};
vkllm_tensor_new(context, "input", in_shapes, vkllm_dtype_float16,
                 VKLLM_OP_NONE, NULL, 0, NULL, 0, true, &input);

// Create embedding operation
struct vkllm_tensor *embedded;
struct vkllm_tensor *embed_weight; // Load your weights
uint32_t emb_shapes[] = {1, 1, seq_len, embed_dim};
vkllm_tensor_new(context, "embedded", emb_shapes, vkllm_dtype_float16,
                 VKLLM_OP_EMBEDDING, &embed_weight, 1, 
                 &embed_params, sizeof(params), false, &embedded);

// Add to graph
vkllm_graph_add_input(context, graph, input);
vkllm_graph_add_node(context, graph, embedded);
```

### Running GGUF Models

The library includes a complete LLaMA 2 implementation:

```bash
# Download a GGUF model (example)
wget https://huggingface.co/models/llama-2-7b-gguf

# Run inference test
bazel build //tests:vkllm_test_infer_gguf
./bazel-bin/tests/vkllm_test_infer_gguf path/to/model.gguf
```

## Testing

### Run All Tests
```bash
./run_all_tests.sh
```

### Individual Tests
```bash
# Test basic operations
bazel test //tests:vkllm_test_op_matmul
bazel test //tests:vkllm_test_op_embedding
bazel test //tests:vkllm_test_op_rmsnorm
bazel test //tests:vkllm_test_op_softmax
bazel test //tests:vkllm_test_op_rope
bazel test //tests:vkllm_test_op_bin

# Test tensor operations
bazel test //tests:vkllm_test_tensor_reshape
bazel test //tests:vkllm_test_op_copy
bazel test //tests:vkllm_test_op_permute

# Test data transfer
bazel test //tests:vkllm_test_transfer

# Test graph and data structures
bazel test //tests:vkllm_test_hashset
bazel test //tests:vkllm_test_hashset_perf
```

## Documentation

Additional documentation is available in the `docs/` directory:

- [MatMul Operation Explanation](docs/matmul_explanation.md) - Detailed guide on matrix multiplication
- [MatMul Visual Example](docs/matmul_visual_example.md) - Visual examples of MatMul operations
- [Computational Graph Functions](docs/vkllm_graph_functions.md) - Graph API reference
- [HashSet Implementation](docs/vkllm_hashset.md) - Internal data structure documentation

## Project Structure

```
vkllm.c/
├── src/
│   ├── core/           # Core library (context, tensor, graph, ops)
│   ├── models/         # Model implementations (LLaMA 2)
│   └── shaders/        # Vulkan compute shaders
├── tests/              # Unit and integration tests
├── docs/               # Additional documentation
├── scripts/            # Utility scripts
├── thirdparty/         # Third-party dependencies
├── vulkan_rules/       # Bazel rules for Vulkan
└── tools/              # Build tools and utilities
```

## Performance Tips

1. **Use FP16**: Enable float16 precision for faster inference with minimal accuracy loss
2. **Batch Processing**: Process multiple sequences together when possible
3. **Memory Management**: Reuse tensors and graphs across inference iterations
4. **Pipeline Caching**: The library caches compiled shaders for faster subsequent runs

## Troubleshooting

### Vulkan Device Not Found
```bash
# Check Vulkan installation
vulkaninfo

# List available GPUs
vulkaninfo --summary
```

### Build Errors
```bash
# Clean and rebuild
bazel clean
bazel build //...
```

### Memory Issues
- Reduce batch size
- Use FP16 instead of FP32
- Free tensors and graphs when no longer needed

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style
- Follow the existing `.clang-format` configuration
- Run `clang-format` before committing
- Add tests for new features
- Update documentation as needed

## Roadmap

- [ ] KV-cache support
- [ ] Quantization support (INT8, INT4)
- [ ] Support for more model architectures (GPT, DeepSeek, etc.)
- [ ] Multi-GPU support

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Vulkan SDK by Khronos Group
- VulkanMemoryAllocator library
- GGUF format and tools
- Check framework for unit testing

## Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: See `docs/` directory for detailed guides
- **Examples**: Check `tests/` directory for usage examples

---

**Note**: This library is under active development. APIs may change between versions.
