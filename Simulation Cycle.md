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