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
[Image](./figures/Arch_Hier.png)



***

You can use the [editor on GitHub](https://github.com/apuaaChen/accel_sim_code_study.github.io/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/apuaaChen/accel_sim_code_study.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
