---
sort: 7
---
# Operand Collector (OC)

![Image](./figures/OC.PNG)

## Operand Collector Based Register File Unit

The SM core has a single Operand Collector Based Register File Unit modeled by class `opndcoll_rfu_t` as follows
```c++
// shader_core_ctx
opndcoll_rfu_t m_operand_collector;

// opndcoll_rdu_t: operand collector based register file unit
arbiter_t m_arbiter;
typedef std::map<unsigned, std::vector<collector_unit_t> cu_sets_t;
cu_sets_t m_cus;
std::vector<collector_unit_t *> m_cu;
std::vector<input_port_t> m_in_ports;
std::vector<dispatch_unit_t> m_dispatch_units;
```

The unit includes 
* Arbitrator (`arbiter_t`): 