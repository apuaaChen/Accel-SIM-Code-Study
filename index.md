# Accel-SIM Code Study
Zhaodong Chen

This note aims to put together major components from the simulator and help understanding the entire simulation process from the source code level. We will focus on the trace-driven simulation.

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

It include two types of commands: `MemcpyHtoD` and `kernel launch`. The `MemcpyHtoD` is simply defined with a string. The `kernel launch` leads the parser to the `kernel-x.traceg` file. Notably, the `MemcpyHtoD` should match the memory address used in the kernel's trace.

Then, let's see an example of `kernel-x.traceg` file
```c++
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
