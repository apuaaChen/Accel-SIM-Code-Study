---
sort: 3
---
# Issue a Thread Block
When a thread block is issued, all its traces are loaded into the hardware warps of the target SM core. 

## Hardware Warp

Each SM core has a set of hardware warps modeled by class `trace_shd_warp_t`. The number of hardware warps equals to the maximum number of thread supported by the SM divided by the warp size. For instance, the V100 GPU supports 2048 threads per SM core, and the warp size is 32. So the total number of hardware warps equal to 64.

### Definition
The `trace_shd_warp_t` is defined as follows
```c++
class trace_shd_warp_t : public shd_warp_t {
public:
  trace_shd_warp_t(class shader_core_ctx *shader, unsigned warp_size)
      : shd_warp_t(shader, warp_size) {
    trace_pc = 0;
    m_kernel_info = NULL;
  }
  // container of the traces of the warp
  std::vector<inst_trace_t> warp_traces;
  const trace_warp_inst_t *get_next_trace_inst();
  void clear();
  bool trace_done();
  address_type get_start_trace_pc();
  virtual address_type get_pc();
  trace_warp_inst_t *set_kernel(trace_kernel_info_t *kernel_info) {
    m_kernel_info = kernel_info;
  }

private:
  unsigned trace_pc;
  // 
  trace_kernel_info_t *m_kernel_info;
};
```
The hardware warp contains a member `std::vector<inst_trace_t> warp_traces;`. It is the container of the traces of the warp.

***

## Issue a Thread Block

Issuing a thread block takes a hierarchical call as follows:

*  `gpgpu_sim::cycle()`
    * `gpgpu_sim::issue_block2core()`
        * `simt_core_cluster::issue_block2core()`
            * `shader_core_ctx::issue_block2core()`
                * `trace_shader_core_ctx::init_warps()`
                    * `trace_shader_core_ctx::init_traces()`
                        * `trace_kernel_info_t::get_next_threadblock_traces()`

### gpgpu_sim::cycle()
In each simulation cycle, the `gpgpu_sim::cycle()` is called. This function takes no arguments.
```c++
// main()
if (m_gpgpu_sim->active()) {
  m_gpgpu_sim->cycle();
  sim_cycles = true;
  m_gpgpu_sim->deadlock_check();
} 
```

### gpgpu_sim::issue_block2core()
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

### simt_core_cluster::issue_block2core()

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

### shader_core_ctx::issue_block2core()

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

### trace_shader_core_ctx::init_warps()

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
It compute the start hardware warp and end hardware warp of the CTA, and call the `trace_shader_core_ctx::init_traces()`

### trace_shader_core_ctx::init_traces()
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
The `trace_shd_warp_t`s are the hardware warps. They have a vector `trace_shd_warp_t::warp_trace`  to store the traces of the warp. The `warp_trace` of the hardware warps assigned to the thread block are collected into a vector, which is the input of the `trace_kernel_info_t`. At last, the `trace_kernel_info_t::get_next_threadblock_trace()` is involked to fill the container. 