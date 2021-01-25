Zhaodong Chen

This note aims to put together major components from the simulator and help understanding the entire simulation process from the source code level. We will focus on the trace-driven simulation.
***

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

## Class: trace_kernel_info

The `(trace_)kernel_info` is an important interface between the trace files and the performance simulator. In particular, it provides a function that load the trace of a thread block into a vector of vector of `inst_trace_t`:
```c++
class trace_kernel_info_t : public kernel_info_t {
public:
  bool get_next_threadblock_traces(
      std::vector<std::vector<inst_trace_t> *> threadblock_traces);
private:
  // Some other functions and basic-information members
};
```
Most of the members in this class are just providing some functionality and basic informations of a kernel. The most important function is `get_next_threadblock_traces`. It takes a vector of vector as input
```c++
// Input: vector of vector of inst_trace_t
std::vector<std::vector<inst_trace_t> *> threadblock_traces;
```

It is a vector of vector because each thread block has several warps, and each warp has several instructions. So the first level is the index to the warp, and the second level is the index to individual instructions. 

The whole container is cleared at the begining with
```c++
for (unsigned i = 0; i < threadblock_traces.size(); ++i) {
    threadblock_traces[i]->clear();
}
```
Then, the slots are filled with
```c++
threadblock_traces[warp_id]->at(inst_count).parse_from_string(line, trace_version);
```
In which the vector of the target warp is located (`warp_id`), then the slot for the instruction is found (`inst_count`). The slot is filled with the trace by calling the function `inst_trace_t::parse_from_string()`, which gets the following informations from the trace string and encodes it into an instance of class `inst_trace_t`.
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
# Architecture Hierarchy
![./figures/Arch_Hier.png](./figures/Arch_Hier.PNG)

The GPU is modeled as follows. The class `trace_simt_core_cluster` models the whole GPU. Its member, `trace_gpgpu_sim->m_cluster`, is a vector of `trace_simt_core_cluster` objects. Each object models an SM cluster. In each cluster, there is memeber, `trace_simt_core_cluster->m_core`, that models the SM cores in this cluster. Each SM core is modeled with an object of `trace_shader_core_ctx`. Within the SM core, there is a set of hardware warps, `trace_shader_core_ctx->m_warp`, modeled by class `trace_shd_warp_t`. The number of hardware warps in each SM equals to the maximum number of threads (2048 for V100) divided by the warp size 32.

# Issue a Thread Block
When a thread block is issued, all its traces are loaded into the hardware warps of the target SM core. Issuing a thread block takes a hierarchical call as follows:

*  `gpgpu_sim::cycle()`
    * `gpgpu_sim::issue_block2core()`
        * `simt_core_cluster::issue_block2core()`
            * `shader_core_ctx::issue_block2core()`
                * `trace_shader_core_ctx::init_warps()`
                    * `trace_shader_core_ctx::init_traces()`
                        * `trace_kernel_info_t::get_next_threadblock_traces()`

## gpgpu_sim::cycle()
In each simulation cycle, the `gpgpu_sim::cycle()` is called. This function takes no arguments.
```c++
// main()
if (m_gpgpu_sim->active()) {
  m_gpgpu_sim->cycle();
  sim_cycles = true;
  m_gpgpu_sim->deadlock_check();
} 
```

## gpgpu_sim::issue_block2core()
In the `gpgpu_sim::cycle()`, besides calling `cycle()` function of other units like the cores, it also call the function `gpgpu_sim::issue_block2core()`, which also takes no argument.
```c++
// gpgpu_sim::cycle()
issue_block2core();
```
In `gpgpu_sim::issue_block2core()`, we have
```c++
void gpgpu_sim::issue_block2core() {
  unsigned last_issued = m_last_cluster_issue;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    // starting from the previous issued SM cluster, call issue_block2core one by one
    unsigned idx = (i + last_issued + 1) % m_shader_config->n_simt_clusters;
    // Multiple CTAs (thread blocks) can be issued in each step. 
    // The function simt_core_cluster::issue_block2core() returns the number of CTAs issued
    unsigned num = m_cluster[idx]->issue_block2core();
    if (num) {
      m_last_cluster_issue = idx;
      // increment the launched CTAs
      m_total_cta_launched += num;
    }
  }
}
```
Basically, all the SM clusters are traversed. The traversal starts from the last issued cluster. For each cluster, `issue_block2core` is called, which returns the number of blocks issued by this cluster. This is incremented to the member `gpgpu_sim::m_total_cta_launched`.

## simt_core_cluster::issue_block2core()

In `simt_core_cluster::issue_block2core()`, we have
```c++
unsigned simt_core_cluster::issue_block2core() {
  // a counter initialized to 0
  unsigned num_blocks_issued = 0;
  // Traverse all the SMs in this SM cluster
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    // Also start from the m_cta_issue_next_core
    unsigned core =
        (i + m_cta_issue_next_core + 1) % m_config->n_simt_cores_per_cluster;
		
    // Fetch the kernel
    kernel_info_t *kernel;
    // Something about kernel selection
    
    // If there are remaining CTAs in the kernel and the core can issue a block
    if (m_gpu->kernel_more_cta_left(kernel) && m_core[core]->can_issue_1block(*kernel)) {
      // issue the block
      m_core[core]->issue_block2core(*kernel);
      // issue one CTA (block) to a core at each cycle
      num_blocks_issued++;
      // last issued core is the current core
      m_cta_issue_next_core = core;
      break;
    }
  }
  return num_blocks_issued;
}
```
Each SM cluster sweeps all its SMs. If there are unissued blocks of the kernel and the SM can issue 1 block, a block is issued to the core. Let's take a step aside and see how each SM determines whether it can issue more blocks
```c++
bool shader_core_ctx::can_issue_1block(kernel_info_t &kernel) {
  // Something about concurrent kernels on one SM
  return (get_n_active_cta() < m_config->max_cta(kernel));
}
```
This is quite simple. As the config knows the resources (reg, shared memory) occupied by the kernel, it can compute the maximum number of CTAs supported by each SM. So simply check whether the number of active CTAs is smaller than the upper bound.

## shader_core_ctx::issue_block2core()

In `shader_core_ctx::issue_block2core()`, we only want to issue 1 CTA if it is possible. We have
```c++
void shader_core_ctx::issue_block2core(kernel_info_t &kernel) {

  // find a free CTA context
  // init it with the max value
  unsigned free_cta_hw_id = (unsigned)-1;

  // get the maximum number of CTAs supported by the SM
  unsigned max_cta_per_core;
  max_cta_per_core = kernel_max_cta_per_shader;
  
  // Find an empty slot
  for (unsigned i = 0; i < max_cta_per_core; i++) {
    if (m_cta_status[i] == 0) {
      free_cta_hw_id = i;
      break;
    }
  }
  assert(free_cta_hw_id != (unsigned)-1); // the free_cta_hw_id should be updated

  // determine hardware threads and warps that will be used for this CTA
  // Get the thread block size. 
  int cta_size = kernel.threads_per_cta();
  
  int padded_cta_size = cta_size;
  // Something about padding the size when it is not multiple of warp size
  

  // compute the target warp id. There are 2048/32 hardware warps. 

  unsigned int start_thread, end_thread;
  
  start_thread = free_cta_hw_id * padded_cta_size;
    end_thread = start_thread + cta_size;

  // reset the microarchitecture state of the selected hardware thread and warp
  // contexts
  reinit(start_thread, end_thread, false);

  // Something about Functional simulation

  // initialize the SIMT stacks and fetch hardware
  init_warps(free_cta_hw_id, start_thread, end_thread, ctaid, cta_size, kernel);
  m_n_active_cta++;
}
```
It first computes start and end threads occupied by the CTA. Then it calls the `trace_shader_core_ctx::init_warps`

## trace_shader_core_ctx::init_warps()

In `trace_shader_core_ctx::init_warps`, we have
```c++
void trace_shader_core_ctx::init_warps(unsigned cta_id, unsigned start_thread,
                                       unsigned end_thread, unsigned ctaid,
                                       int cta_size, kernel_info_t &kernel) {
  // call base class
  shader_core_ctx::init_warps(cta_id, start_thread, end_thread, ctaid, cta_size,
                              kernel);

  // then init traces
  unsigned start_warp = start_thread / m_config->warp_size;
  unsigned end_warp = end_thread / m_config->warp_size +
                      ((end_thread % m_config->warp_size) ? 1 : 0);

  init_traces(start_warp, end_warp, kernel);
}
```
It compute the start warp and end hardware warp of the CTA, and call the `trace_shader_core_ctx::init_traces()`

## trace_shader_core_ctx::init_traces()
```c++
void trace_shader_core_ctx::init_traces(unsigned start_warp, unsigned end_warp,
                                        kernel_info_t &kernel) {
	
  // create the vector of vector of instructions
  // currently, it is empty. Its entries will be the warp_traces member in each trace_shf_warp_t
  // They will be added to the list in the following loop
  std::vector<std::vector<inst_trace_t> *> threadblock_traces;
  
  // this locates the warps used for this kernel (I guess)
  for (unsigned i = start_warp; i < end_warp; ++i) {
    // simple reinterpretation
    trace_shd_warp_t *m_trace_warp = static_cast<trace_shd_warp_t *>(m_warp[i]);
    // clear the warp
    m_trace_warp->clear();
    // the threadblock_traces is composed of the warp_traces member in each trace_shf_warp_t
    threadblock_traces.push_back(&(m_trace_warp->warp_traces));
  }
  // cast input kernel into into trace_kernel_info_t
  trace_kernel_info_t &trace_kernel =
      static_cast<trace_kernel_info_t &>(kernel);
  // fill the threadblock_traces with the with the traces from file
  trace_kernel.get_next_threadblock_traces(threadblock_traces);
  // Something else
}
```
The `trace_shd_warp_t`s are the hardware warps. They have a vector `trace_shd_warp_t::warp_trace`  to store the traces of the warp.
## get_next_threadblock_traces()
```c++
bool trace_kernel_info_t::get_next_threadblock_traces(
   std::vector<std::vector<inst_trace_t> *> threadblock_traces) {
  
  // threadblock_traces is a vector of vector of inst_trace_t
  // the vector of vector is reset to empty containers
  for (unsigned i = 0; i < threadblock_traces.size(); ++i) {
    threadblock_traces[i]->clear();
  }
	
  // The parser is called to get the next_threadblock_trace
  bool success = m_parser->get_next_threadblock_traces(
      threadblock_traces, m_kernel_trace_info->trace_verion);

  return success;
} 

bool trace_parser::get_next_threadblock_traces(
    std::vector<std::vector<inst_trace_t> *> threadblock_traces,
    unsigned trace_version) {
  
  // Simularly, the container is cleared at the begining
  for (unsigned i = 0; i < threadblock_traces.size(); ++i) {
    threadblock_traces[i]->clear();
  }

  unsigned block_id_x = 0, block_id_y = 0, block_id_z = 0;
  // A flag the indicates the screening reaches a threadblock
  bool start_of_tb_stream_found = false;

  unsigned warp_id = 0;
  unsigned insts_num = 0;
  unsigned inst_count = 0;  // counter for the instructions.
	
  // Actually, the file is not closed
  // So I guess it will continue after the loop breaks and accessed again
  while (!ifs.eof()) {
    std::string line;
    std::stringstream ss;
    std::string string1, string2;

    getline(ifs, line);
		// Skip the empty line
    if (line.length() == 0) {
      continue;
    } else {
      ss.str(line);
      ss >> string1 >> string2;
      if (string1 == "#BEGIN_TB") {
        if (!start_of_tb_stream_found) {
          start_of_tb_stream_found = true;  // the flag is set to True
        } else
          assert(0 &&
                 "Parsing error: thread block start before the previous one "
                 "finishes");
      } else if (string1 == "#END_TB") {  // end of the thread blcok
        assert(start_of_tb_stream_found);
        break; // end of TB stream
      } else if (string1 == "thread" && string2 == "block") {
        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "thread block = %d,%d,%d", &block_id_x,
               &block_id_y, &block_id_z);  // Set the threadblock id
        std::cout << line << std::endl;
      } else if (string1 == "warp") {
        // the start of new warp stream
        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "warp = %d", &warp_id);  // Get warp id
      } else if (string1 == "insts") {
        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "insts = %d", &insts_num);  // Get the number of warps
        threadblock_traces[warp_id]->resize(
            insts_num); // allocate all the space at once
        inst_count = 0;
      } else {
        // vector::at: returns a reference to the element at position n in the vector
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
# Simulation Cycle

When the `gpgpu_sim::cycle()` is called, it calls the `cycle` of all its SM clusters as follows
```c++
void gpgpu_sim::cycle() {
  // Something
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
      m_cluster[i]->core_cycle();
    }
  }
}
```
When the `simt_core_cluster::core_cycle()` is called, it calls the `cycle` of all the SM cores in it.
```c++
void simt_core_cluster::core_cycle() {
  for (std::list<unsigned>::iterator it = m_core_sim_order.begin();it != m_core_sim_order.end(); ++it) {
    m_core[*it]->cycle();
  }

  if (m_config->simt_core_sim_order == 1) {
    m_core_sim_order.splice(m_core_sim_order.end(), m_core_sim_order,
                            m_core_sim_order.begin());
  }
}
```
Each SM core has several pipeline stages: `fetch()`, `decode()`, `issue()`, `read_operands()`, `execute()`, and `writeback()`, its `cycle()` function is defined as follows
```c++
// shader_core_ctx::cycle()
void shader_core_ctx::cycle() {
  if (!isactive() && get_not_completed() == 0) return;

  m_stats->shader_cycles[m_sid]++;
  writeback();
  execute();
  read_operands();
  issue();
  for (int i = 0; i < m_config->inst_fetch_throughput; ++i) {
    decode();
    fetch();
  }
}
```

# Instruction Decode (ID)
In this part, we will study how the instruction is fetched.

## Instruction Fetch Buffer
![Image](./figures/Ifetch.PNG)

The Instruction Fetch Buffer (`ifetch_buffer_t`) models the interface between the instruction cache (I-Cache) and the SM core. It is defined as follows
```c++
// shader_core_ctx
ifetch_buffer_t m_inst_fetch_buffer;

struct ifetch_buffer_t {
  ifetch_buffer_t() { m_valid = false; }

  ifetch_buffer_t(address_type pc, unsigned nbytes, unsigned warp_id) {
    m_valid = true;
    m_pc = pc;
    m_nbytes = nbytes;
    m_warp_id = warp_id;
  }

  bool m_valid;
  address_type m_pc;
  unsigned m_nbytes;
  unsigned m_warp_id;
};
```

## 


<details>

* If the `m_inst_buffer` is empty (not valid)
  * If there is an instruction in the instruction cache (it is ready)
    * put the instruction into the `m_inst_buffer`
  * Otherwise if there isn't a ready instruction in the cache
    * traverse all the hardware warps (2048/32). If the warp is functioning, not waiting for instruction cache missing, and its instruction buffer is empty: generate a memory fetch request.
    * Check if the fetch can be directly obtained in the instruction cache.
      * If it is, put the instruction into `m_inst_buffer`
      * otherwise, the hardware warp is set to instruction cache missing state.
* Run the L1 instruction cache cycle.

</details>