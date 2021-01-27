---
sort: 9
---
# Writeback (WB)
![Image](./figures/WB.PNG)

This note covers the `writeback()` stage of the SM core's pipeline.
```c++
// shader_core_ctx::cycle()
writeback();
```