---
sort: 1
---
# Load Trace to GPU

## Input Files

The trace-driven simulation takes two input files: `kernel-x.traceg` and `kernelslist.g`. An example of `kernelslist.g` is as follows

```
MemcpyHtoD,0x00007efe7b500000,2052
MemcpyHtoD,0x00007efe7b500a00,262144
MemcpyHtoD,0x00007efe7b540a00,524288
MemcpyHtoD,0x00007efe7b600000,262144
MemcpyHtoD,0x00007efe7b5c0a00,2048
kernel-1.traceg
```

It includes two types of commands: `MemcpyHtoD` and `kernel launch`. The `MemcpyHtoD` is simply defined with a string. The `kernel launch` leads the parser to the `kernel-x.traceg` file. Notably, the `MemcpyHtoD` should match the memory address used in the kernel's trace.

Then, let's see an example of `kernel-x.traceg` file
```shell
# Basic information of the kernel
-kernel name = KERNEL_NAME
-kernel id = 1
-grid dim = (512,8,1)
-block dim = (32,1,1)
-shmem = 0
-nregs = 48
-binary version = 70  # Used to select the opcode map. E.g. 70 for VOLTA
-cuda stream id = 0
-shmem base_addr = 0x00007efeb0000000
-local mem base_addr = 0x00007efeb2000000
-nvbit version = 1.5
-accelsim tracer version = 3

#traces format = PC mask dest_num [reg_dests] opcode src_num [reg_srcs] mem_width [adrrescompress?] [mem_addresses]



#BEGIN_TB

thread block = 0,0,0

warp = 0
insts = 1212
0000 ffffffff 1 R1 IMAD.MOV.U32 2 R255 R255 0 
0010 00000000 0 SHFL.IDX 4 R255 R255 R255 R255 0 
.....
1020 ffffffff 1 R24 LDG.E.128.CONSTANT.SYS 1 R44 16 2 0x7efe7b60c300 16 16 16 2000 16 16 16 976 16 16 16 2000 16 16 16 6096 16 16 16 4048 16 16 16 3024 16 16 16 976 16 16 16 
.....

warp = 1
inst = ...
....

#END_TB

#BEGIN_TB
...
#END_TB

..
```

First, the basic information of the kernel is provided. Then, the traces for each thread block is marked with key words `#BEGIN_TB` and `#END_TB`. Each thread block has several warps, the begining of each warp is marked with `warp = warp_id` followed by the total number of traces of the warp (e.g. `insts = 1212`).

The format of the trace is as follows
```
PC mask dest_num [reg_dests] opcode src_num [reg_srcs] mem_width [adrrescompress?] [mem_addresses]
```
## Process the Trace Files
In the main function, the trace files are processed as follows
```c++
// Step 1: create the trace_parser
trace_parser tracer(tconfig.get_traces_filename());

// Step 2: get all the MemcpyHtoD and kernels (commands)
std::vector<trace_command> commandlist = tracer.parse_commandlist_file();

// Loop: travers all the commands
for command in commandlist{
    if command is MemcpyHtoD{
        // Do something
    }
    if command is Launch Kernel{
        // get kernel info
        trace_kernel_info_t kernel_info = create_kernel_info(...);
        // Load the kernel_info into the simulator
        m_gpgpu_sim->launch(kernel_info);
        while (m_gpgpu_sim->active()){
            m_gpgpu_sim->cycle()
        }
    }
}    
```
In the first step, a tracer is created. The tracer gets a list of commands from the `kernelslist.g` file. For each command in the command list, if it is launching a kernel, an object of `trace_kernel_info_t` will be created. It will be used as the interface between the trace file and the performance model.

## Class: trace_kernel_info

The `(trace_)kernel_info` is an important interface between the trace files and the performance simulator. In particular, it provides a function that load the trace of a thread block into a vector of vector of `inst_trace_t`.

### Definition 

The class is defined as follows
```c++
class trace_kernel_info_t : public kernel_info_t {
public:
  trace_kernel_info_t(dim3 gridDim, dim3 blockDim,
                      trace_function_info *m_function_info,
                      trace_parser *parser, class trace_config *config,
                      kernel_trace_t *kernel_trace_info);

  bool get_next_threadblock_traces(
      std::vector<std::vector<inst_trace_t> *> threadblock_traces);

private:
  trace_config *m_tconfig;
  const std::unordered_map<std::string, OpcodeChar> *OpcodeMap;
  trace_parser *m_parser;
  kernel_trace_t *m_kernel_trace_info;

  friend class trace_shd_warp_t;
};
```
Most of the members in this class are just providing some functionality and basic informations of a kernel. The most important function is `get_next_threadblock_traces`. 

### Init()
```c++
// trace_kernel_info_t *create_kernel_info()
trace_kernel_info_t *kernel_info = new trace_kernel_info_t(gridDim, blockDim, function_info, parser, config, kernel_trace_info);

trace_kernel_info_t::trace_kernel_info_t(dim3 gridDim, dim3 blockDim,
                                         trace_function_info *m_function_info,
                                         trace_parser *parser,
                                         class trace_config *config,
                                         kernel_trace_t *kernel_trace_info)
    : kernel_info_t(gridDim, blockDim, m_function_info) {
  m_parser = parser;
  m_tconfig = config;
  m_kernel_trace_info = kernel_trace_info;

  // resolve the binary version
  if (kernel_trace_info->binary_verion == VOLTA_BINART_VERSION)
    OpcodeMap = &Volta_OpcodeMap;
  else if (kernel_trace_info->binary_verion == PASCAL_TITANX_BINART_VERSION ||
           kernel_trace_info->binary_verion == PASCAL_P100_BINART_VERSION)
    OpcodeMap = &Pascal_OpcodeMap;
  else if (kernel_trace_info->binary_verion == KEPLER_BINART_VERSION)
    OpcodeMap = &Kepler_OpcodeMap;
  else if (kernel_trace_info->binary_verion == TURING_BINART_VERSION)
    OpcodeMap = &Turing_OpcodeMap;
  else {
    printf("unsupported binary version: %d\n",
           kernel_trace_info->binary_verion);
    fflush(stdout);
    exit(0);
  }
}
```

The `trace_kernel_info_t` contains information about the kernel like grad/block dim and opcode map. The `init()` function simply set the members in the object.

### get_next_threadblock_traces()
The function is defined as follows
```c++
bool trace_kernel_info_t::get_next_threadblock_traces(
    std::vector<std::vector<inst_trace_t> *> threadblock_traces) {
  // Step 1: clear the container
  for (unsigned i = 0; i < threadblock_traces.size(); ++i) {
    threadblock_traces[i]->clear();
  }
  
  // get the next threadblock traces
  bool success = m_parser->get_next_threadblock_traces(
      threadblock_traces, m_kernel_trace_info->trace_verion);

  return success;
}
```

It takes a vector of vector as input
```c++
// void trace_shader_core_ctx::init_traces()
// Input: vector of vector of inst_trace_t
std::vector<std::vector<inst_trace_t> *> threadblock_traces;
```
It is a vector of vector because each thread block has several warps, and each warp has several instructions. So the first level is the index to the warp, and the second level is the index to individual traces.

In the first step, the context in the `treadblock_traces` is cleared as they belong to the previous thread block. Then, the `m_parser->get_next_threadblock_traces()` is called. 
```c++
bool trace_parser::get_next_threadblock_traces(
    std::vector<std::vector<inst_trace_t> *> threadblock_traces,
    unsigned trace_version) {
  // Step 1: clear the container
  for (unsigned i = 0; i < threadblock_traces.size(); ++i) {
    threadblock_traces[i]->clear();
  }

  unsigned block_id_x = 0, block_id_y = 0, block_id_z = 0;
  bool start_of_tb_stream_found = false;

  unsigned warp_id = 0;
  unsigned insts_num = 0;
  unsigned inst_count = 0;

  while (!ifs.eof()) {
    std::string line;
    std::stringstream ss;
    std::string string1, string2;

    getline(ifs, line);

    if (line.length() == 0) {
      continue;
    } else {
      ss.str(line);
      ss >> string1 >> string2;
      // Reach the begining of the thread block
      if (string1 == "#BEGIN_TB") {
        if (!start_of_tb_stream_found) {
          start_of_tb_stream_found = true;
        } else
          assert(0 &&
                 "Parsing error: thread block start before the previous one "
                 "finishes");
      } 
      // Reach the end of the thread block
      else if (string1 == "#END_TB") {
        assert(start_of_tb_stream_found);
        break; // end of TB stream
      } 
      // The following lines process the thread block index
      // warp id, and total number of instructions
      else if (string1 == "thread" && string2 == "block") {
        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "thread block = %d,%d,%d", &block_id_x,
               &block_id_y, &block_id_z);
        std::cout << line << std::endl;
      } else if (string1 == "warp") {
        // the start of new warp stream
        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "warp = %d", &warp_id);
      } else if (string1 == "insts") {
        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "insts = %d", &insts_num);
        threadblock_traces[warp_id]->resize(
            insts_num); // allocate all the space at once
        inst_count = 0;
      } 
      // The line is a trace
      else {
        assert(start_of_tb_stream_found);
        threadblock_traces[warp_id]
            ->at(inst_count)
            .parse_from_string(line, trace_version);
        inst_count++;
      }
    }
  }

  return true;
}
```
At first, the vector of the target warp is located (`warp_id`), then the slot for the instruction is found (`inst_count`).
When the input line is a trace, the `inst_trace_t::parse_from_string` is called. The `inst_trace_t` models a single instruction trace, which is defined as follows
```c++
struct inst_trace_t {
  inst_trace_t();
  inst_trace_t(const inst_trace_t &b);
	
  // Basic informations
  unsigned m_pc;  // pc of the instruction
  unsigned mask;  // active mask
  unsigned reg_dsts_num;  // number of destinition register
  unsigned reg_dest[MAX_DST];  // an array of destinition register
  std::string opcode;  // Opcode string
  unsigned reg_srcs_num;  // number of src registers
  unsigned reg_src[MAX_SRC];  // an array of source register
  inst_memadd_info_t *memadd_info;  // memory info
	
  // Other helper functions
  bool parse_from_string(std::string trace, unsigned tracer_version);

  bool check_opcode_contain(const std::vector<std::string> &opcode,
                            std::string param) const;

  unsigned
  get_datawidth_from_opcode(const std::vector<std::string> &opcode) const;

  std::vector<std::string> get_opcode_tokens() const;

  ~inst_trace_t();
};
```
The `inst_trace_t::parse_from_string` simply fills the member's value in the struct.

## Load to GPU

```c++
// main()
trace_kernel_info_t kernel_info = create_kernel_info(...);
m_gpgpu_sim->launch(kernel_info);

// class gpgpu_sim
std::vector<kernel_info_t *> m_running_kernels;

void gpgpu_sim::launch(kernel_info_t *kinfo) {
  unsigned cta_size = kinfo->threads_per_cta();
  unsigned n = 0;
  for (n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done()) {
      m_running_kernels[n] = kinfo;
      break;
    }
  }
  assert(n < m_running_kernels.size());
}

```
The GPU (`gpgpu_sim`) contains a vector of `kernel_info_t` that stores the `trace_kernel_info_t` objects of the launched kernels. After creating the object, it is appended into the list `gpgpu_sim::m_running_kernels`.
